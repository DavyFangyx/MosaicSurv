import os
import copy
from pathlib import Path
import torch
import joblib
import random
import math
import sys
import argparse
import importlib.util
import numpy as np
import pandas as pd
import torch.nn as nn
import time as sys_time
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.model_selection import KFold 
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index as ci
from sklearn.model_selection import StratifiedKFold
from mae_model import fusion_model_mae_2
from util import Logger, get_patients_information,get_all_ci,get_val_ci,adjust_learning_rate
from mae_utils import generate_mask

ROOT_DIR = Path(__file__).resolve().parents[5]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from missing_baseline_result_utils import build_results_root, build_seed_dir, build_fold_dir, write_experiment_file
from utils.missing_mask_protocol import avail_to_hgcn_in_mask, load_fold_mask_lookup, unified_mask_csv_path
from models.missing_modality_baselines.common import (
    ALL_EVAL_SUBSETS,
    hgcn_use_type_for_subset,
    parse_eval_modalities,
    write_eval_subset_outputs,
)



def _requested_eval_subsets(args):
    return parse_eval_modalities(getattr(args, "eval_modalities", "off"))


def _should_write_eval_subsets(args):
    return bool(parse_eval_modalities(getattr(args, "eval_modalities", "off")))


HGCN_FOLD_METRIC_COLUMNS = [
    "test_cindex",
    "test_cindex_ipcw",
    "test_IBS",
    "test_iauc",
    "test_iauc_list",
    "test_loss",
    "test_BS",
]


def _hgcn_fold_metric_row(test_cindex):
    return {
        "test_cindex": test_cindex,
        "test_cindex_ipcw": 0.0,
        "test_IBS": 0.0,
        "test_iauc": 0.0,
        "test_iauc_list": 0.0,
        "test_loss": 0.0,
        "test_BS": 0.0,
    }


def _append_hgcn_subset_row(rows, fold, subset, test_cindex):
    rows.append({
        "fold": int(fold),
        "subset": subset,
        **_hgcn_fold_metric_row(test_cindex),
    })


def _write_hgcn_fold_test_result(results_root, rows):
    df = pd.DataFrame(rows, columns=HGCN_FOLD_METRIC_COLUMNS)
    df.to_csv(Path(results_root) / "test_result.csv")


def _load_local(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HGCN_BASELINE_DIR = ROOT_DIR / "models" / "missing_modality_baselines"
_hgcn_graph_build = _load_local(
    "survpgc_hgcn_graph_build",
    _HGCN_BASELINE_DIR / "hgcn_graph_build.py",
)
_hgcn_paths = _load_local(
    "survpgc_hgcn_paths",
    _HGCN_BASELINE_DIR / "hgcn_paths.py",
)
load_hgcn_graphs_from_dirs = _hgcn_graph_build.load_hgcn_graphs_from_dirs
hgcn_pack_complete = _hgcn_paths.hgcn_pack_complete
remap_workspace_path_to_hgcn_data = _hgcn_paths.remap_workspace_path_to_hgcn_data


def _infer_hgcn_in_feats(all_data, field, default):
    for data in all_data.values():
        x = getattr(data, field, None)
        if x is None:
            continue
        shape = getattr(x, "shape", None)
        if shape is not None and len(shape) >= 2 and int(shape[-1]) > 0:
            return int(shape[-1])
    return int(default)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _resolve_split_dir(args, cancer_type):
    if getattr(args, "split_dir", None):
        return args.split_dir
    return os.path.join(args.split_root, f"tcga_{cancer_type.lower()}")


def _normalize_study_name(cancer_type):
    study = str(cancer_type).strip()
    if not study:
        raise ValueError("cancer_type is required")
    if not study.startswith("tcga_"):
        study = f"tcga_{study}"
    return study


def _load_split_csv(split_csv_path):
    split_df = pd.read_csv(split_csv_path)
    first_col = str(split_df.columns[0]) if len(split_df.columns) else ""
    if first_col.startswith("Unnamed"):
        split_df = split_df.drop(columns=split_df.columns[0])
    missing = [col for col in ("train", "val", "test") if col not in split_df.columns]
    if missing:
        raise ValueError(f"{split_csv_path} missing columns: {missing}")
    return [
        split_df[col].dropna().astype(str).tolist()
        for col in ("train", "val", "test")
    ]


def _normalize_hgcn_case_id(case_id):
    return str(case_id).strip().upper()[:12]


def _intersect_split_ids(split_ids, available_ids, split_name, split_csv_path):
    available = set(str(x) for x in available_ids)
    kept = [str(case_id) for case_id in split_ids if str(case_id) in available]
    missing = [str(case_id) for case_id in split_ids if str(case_id) not in available]
    if not kept:
        raise ValueError(
            f"{split_csv_path} {split_name} split has 0 cases after intersecting with assembled HGCN graphs"
        )
    if missing:
        print(
            f"[HGCN] drop {len(missing)} {split_name} ids missing from assembled graphs; kept={len(kept)}",
            flush=True,
        )
    return kept


def _hgcn_in_mask_for_case(args, case_id, *, training):
    missing_mode = getattr(args, "missing_mode", "model_gen")
    if missing_mode == "unified_mask_csv":
        lookup = getattr(args, "_unified_mask_lookup", None)
        if lookup is None:
            raise ValueError("unified_mask_csv requires a loaded fold mask lookup.")
        key = _normalize_hgcn_case_id(case_id)
        try:
            avail = lookup[key]
        except KeyError as exc:
            raise KeyError(f"Case {key!r} is missing from the unified HGCN mask csv.") from exc
        return avail_to_hgcn_in_mask(avail, args.train_use_type)
    if training:
        return generate_mask(num=len(args.train_use_type))
    return []



def prediction(all_data,v_model,val_id,patient_and_time,patient_sur_type,args):
    v_model.eval()
       
    lbl_pred_all = None
    status_all = []
    survtime_all = []
    val_pre_time = {}
    val_pre_time_img = {}
    val_pre_time_rna = {}
    val_pre_time_cli = {}
    iter = 0
    
    with torch.no_grad():
        for i_batch, id in enumerate(val_id):

            graph = all_data[id].to(device)
            if args.train_use_type != None:
                use_type_eopch = args.train_use_type
            else:
                use_type_eopch = graph.data_type
            mask = _hgcn_in_mask_for_case(args, id, training=False)
            out_pre,out_fea,out_att,_ = v_model(graph,args.train_use_type,use_type_eopch,mask,mix=args.mix)
            lbl_pred = out_pre[0]

            survtime_all.append(patient_and_time[id])
            status_all.append(patient_sur_type[id])

            val_pre_time[id] = lbl_pred.cpu().detach().numpy()[0]

            if iter == 0 or lbl_pred_all == None:
                lbl_pred_all = lbl_pred
            else:
                lbl_pred_all = torch.cat([lbl_pred_all, lbl_pred])

            iter += 1
            
            if 'img' in use_type_eopch:
                val_pre_time_img[id] = out_pre[1][use_type_eopch.index('img')].cpu().detach().numpy()
            if 'rna' in use_type_eopch:
                val_pre_time_rna[id] = out_pre[1][use_type_eopch.index('rna')].cpu().detach().numpy()            
            if 'cli' in use_type_eopch:
                val_pre_time_cli[id] = out_pre[1][use_type_eopch.index('cli')].cpu().detach().numpy()            
            
    survtime_all = np.asarray(survtime_all)
    status_all = np.asarray(status_all)
#     print(lbl_pred_all,survtime_all,status_all)
    loss_surv = _neg_partial_log(lbl_pred_all, survtime_all, status_all)
    loss = loss_surv

    val_ci_ = get_val_ci(val_pre_time,patient_and_time,patient_sur_type)
    val_ci_img_ = 0 
    val_ci_rna_ = 0 
    val_ci_cli_ = 0

    if 'img' in args.train_use_type :
        val_ci_img_ = get_val_ci(val_pre_time_img,patient_and_time,patient_sur_type)
    if 'rna' in args.train_use_type :
        val_ci_rna_ = get_val_ci(val_pre_time_rna,patient_and_time,patient_sur_type)
    if 'cli' in args.train_use_type :
        val_ci_cli_ = get_val_ci(val_pre_time_cli,patient_and_time,patient_sur_type)
    return loss.item(), val_ci_, val_ci_img_, val_ci_rna_, val_ci_cli_
    
        
def _neg_partial_log(prediction, T, E):

    current_batch_len = len(prediction)
    R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_matrix_train[i, j] = T[j] >= T[i]

    train_R = torch.FloatTensor(R_matrix_train)
    train_R = train_R.cuda()

    train_ystatus = torch.tensor(np.array(E),dtype=torch.float).to(device)

    theta = prediction.reshape(-1)

    exp_theta = torch.exp(theta)
    loss_nn = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus)

    return loss_nn 



def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def train_a_epoch(model,train_data,all_data,patient_and_time,patient_sur_type,batch_size,optimizer,epoch,format_of_coxloss,args):
    model.train() 


    lbl_pred_each = None
    lbl_pred_img_each = None
    lbl_pred_rna_each = None
    lbl_pred_cli_each = None
    batch_feature = None

    survtime_all = []
    status_all = []
    survtime_img = []
    status_img = []    
    survtime_rna = []
    status_rna = []      
    survtime_cli = []
    status_cli = []  
    
    iter = 0
    loss_nn_all = [] 
    train_pre_time = {}
    train_pre_time_img = {}
    train_pre_time_rna = {}
    train_pre_time_cli = {}
    
    all_loss = 0.0 
    mes_loss_of_mae = nn.MSELoss()

    
    mse_loss_of_mae = 0.0
    loss_surv = 0.0
    all_loss_surv = 0.0
    img_loss_surv = 0.0
    rna_loss_surv = 0.0
    cli_loss_surv = 0.0
    for i_batch,id in enumerate(train_data):
        
        iter += 1 
        num_of_model = len(all_data[id].data_type)
        mask = _hgcn_in_mask_for_case(args, id, training=True)
        
        if len(args.train_use_type) == 1:
            assert args.format_of_coxloss == 'one' and args.add_mse_loss_of_mae == False
            if args.train_use_type[0] in all_data[id].data_type:
                graph = all_data[id].to(device)
                out_pre,out_fea,out_att,fea_dict = model(graph,args.train_use_type,args.train_use_type,mix=args.mix) 
                lbl_pred = out_pre[0]
                use_type_eopch = args.train_use_type
                num_of_model = 1
        else:
            if args.train_use_type!=None:
                use_type_eopch = args.train_use_type
                num_of_model = len(use_type_eopch)                
            else:
                use_type_eopch = all_data[id].data_type
            graph = all_data[id].to(device)
            out_pre,out_fea,out_att,fea_dict = model(graph,use_type_eopch,use_type_eopch,mask,mix=args.mix)
            lbl_pred = out_pre[0]

        if len(args.train_use_type) == 1 and args.train_use_type[0] not in all_data[id].data_type:
            pass
        else:
            train_pre_time[id] = lbl_pred.cpu().detach().numpy()


                        
            if args.add_mse_loss_of_mae:
                 mask_bool = torch.as_tensor(mask, dtype=torch.bool, device=device).view(-1)
                 if bool(mask_bool.any()):
                     mse_loss_of_mae += args.mse_loss_of_mae_factor * mes_loss_of_mae(
                         input=fea_dict['mae_out'][mask_bool],
                         target=fea_dict['mae_labels'][mask_bool]
                     )

            survtime_all.append(patient_and_time[id])
            status_all.append(patient_sur_type[id])
            if iter == 0 or lbl_pred_each == None:
                lbl_pred_each = lbl_pred
            else:
                lbl_pred_each = torch.cat([lbl_pred_each, lbl_pred])

            if 'img' in use_type_eopch and len(args.train_use_type) != 1:
                train_pre_time_img[id] = out_pre[1][use_type_eopch.index('img')].cpu().detach().numpy()
                survtime_img.append(patient_and_time[id])
                status_img.append(patient_sur_type[id])            
                if lbl_pred_img_each == None :
                    lbl_pred_img_each = out_pre[1][use_type_eopch.index('img')]
                else:
                    lbl_pred_img_each = torch.cat([lbl_pred_img_each, out_pre[1][use_type_eopch.index('img')]])
            if 'rna' in use_type_eopch and len(args.train_use_type) != 1:
                train_pre_time_rna[id] = out_pre[1][use_type_eopch.index('rna')].cpu().detach().numpy()
                survtime_rna.append(patient_and_time[id])
                status_rna.append(patient_sur_type[id])            
                if lbl_pred_rna_each == None :
                    lbl_pred_rna_each = out_pre[1][use_type_eopch.index('rna')]
                else:
                    lbl_pred_rna_each = torch.cat([lbl_pred_rna_each, out_pre[1][use_type_eopch.index('rna')]])            
            if 'cli' in use_type_eopch and len(args.train_use_type) != 1:
                train_pre_time_cli[id] = out_pre[1][use_type_eopch.index('cli')].cpu().detach().numpy()
                survtime_cli.append(patient_and_time[id])
                status_cli.append(patient_sur_type[id])            
                if lbl_pred_cli_each == None :
                    lbl_pred_cli_each = out_pre[1][use_type_eopch.index('cli')]
                else:
                    lbl_pred_cli_each = torch.cat([lbl_pred_cli_each, out_pre[1][use_type_eopch.index('cli')]])


        if iter % batch_size == 0 or i_batch == len(train_data)-1:

            survtime_all = np.asarray(survtime_all)
            status_all = np.asarray(status_all)

            if np.max(status_all) == 0:
                lbl_pred_each = None
                lbl_pred_img_each = None
                lbl_pred_rna_each = None
                lbl_pred_cli_each = None
                batch_feature = None
                con_loss_label = None
                con_time_label = None
                survtime_all = []
                status_all = []
                survtime_img = []
                status_img = []    
                survtime_rna = []
                status_rna = []      
                survtime_cli = []
                status_cli = [] 
                iter = 0
                mse_loss_of_mae = 0.0
                loss_surv = 0.0
                all_loss_surv = 0.0
                img_loss_surv = 0.0
                rna_loss_surv = 0.0
                cli_loss_surv = 0.0
                continue

            optimizer.zero_grad() 


            if format_of_coxloss == 'one':
                all_loss_surv = _neg_partial_log(lbl_pred_each, survtime_all, status_all)
                loss_surv = args.all_cox_loss_factor * all_loss_surv
            elif format_of_coxloss == 'multi':
                if lbl_pred_img_each != None:
                    img_loss_surv = args.img_cox_loss_factor * _neg_partial_log(lbl_pred_img_each, survtime_img, status_img)
                    loss_surv += img_loss_surv  

                if lbl_pred_rna_each != None:
                    rna_loss_surv = args.rna_cox_loss_factor * _neg_partial_log(lbl_pred_rna_each, survtime_rna, status_rna)
                    loss_surv += rna_loss_surv

                if lbl_pred_cli_each != None:    
                    cli_loss_surv = args.cli_cox_loss_factor * _neg_partial_log(lbl_pred_cli_each, survtime_cli, status_cli)
                    loss_surv += cli_loss_surv 
            else:
                raise("Wrong format_of_coxloss")

            loss = loss_surv   
                
            if args.add_mse_loss_of_mae: 
                mse_loss_of_mae/=iter
                loss += mse_loss_of_mae 

            all_loss += loss.item()
            loss.backward()
            if epoch == 0:
                print('*',end='')
            else:  
                optimizer.step()

            torch.cuda.empty_cache()
            lbl_pred_each = None
            lbl_pred_img_each = None
            lbl_pred_rna_each = None
            lbl_pred_cli_each = None
            batch_feature = None
            con_loss_label = None
            con_time_label = None
            survtime_all = []
            status_all = []
            survtime_img = []
            status_img = []    
            survtime_rna = []
            status_rna = []      
            survtime_cli = []
            status_cli = [] 
            loss_nn_all.append(loss.data.item())
            con_loss = 0.0
            mse_loss = 0.0
            mse_loss_of_mae = 0.0
            kl_loss = 0.0 
            loss_surv = 0.0
            all_loss_surv = 0.0
            img_loss_surv = 0.0
            rna_loss_surv = 0.0
            cli_loss_surv = 0.0
            iter = 0            

    t_train_ci_img = 0
    t_train_ci_rna = 0
    t_train_ci_cli = 0
    all_loss = all_loss/len(train_data)*batch_size
    t_train_ci = get_val_ci(train_pre_time,patient_and_time,patient_sur_type)
    if len(args.train_use_type) != 1:
        if 'img' in args.train_use_type :
            t_train_ci_img = get_val_ci(train_pre_time_img,patient_and_time,patient_sur_type)
        if 'rna' in args.train_use_type :
            t_train_ci_rna = get_val_ci(train_pre_time_rna,patient_and_time,patient_sur_type)
        if 'cli' in args.train_use_type :
            t_train_ci_cli = get_val_ci(train_pre_time_cli,patient_and_time,patient_sur_type)

    return all_loss,t_train_ci,t_train_ci_img,t_train_ci_rna,t_train_ci_cli


def main(args): 
    start_seed = args.start_seed
    cancer_type = args.cancer_type
    if getattr(args, "run_name", "default") == "default":
        args.run_name = f"{cancer_type}__hgcn"
    repeat_num = args.repeat_num
    drop_out_ratio = args.drop_out_ratio
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    details = args.details
    fusion_model = args.fusion_model
    format_of_coxloss = args.format_of_coxloss
    if_adjust_lr = args.if_adjust_lr
    

    label = "{} {} lr_{} {}_coxloss".format(cancer_type, details, lr,format_of_coxloss) 
    
    if args.add_mse_loss_of_mae:
        label = label + " {}*mae_loss".format(args.mse_loss_of_mae_factor)

    if args.img_cox_loss_factor != 1:
        label = label + " img_ft_{}".format(args.img_cox_loss_factor)
    if args.rna_cox_loss_factor != 1:
        label = label + " rna_ft_{}".format(args.rna_cox_loss_factor)    
    if args.cli_cox_loss_factor != 1:
        label = label + " cli_ft_{}".format(args.cli_cox_loss_factor)    
    if args.mix:
        label = label + " mix"
    if args.train_use_type != None:
        label = label + ' use_'
        for x in args.train_use_type:
            label = label + x
        
    
    print(label)                                                                                  

    results_root = build_results_root(args, "hgcn")
    args.results_dir = str(results_root)
    write_experiment_file(results_root, {
        "exp_group": getattr(args, "exp_group", "default"),
        "run_name": getattr(args, "run_name", "default"),
        "study": cancer_type,
        "modality": "hgcn",
        "repeat_num": repeat_num,
        "start_seed": start_seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "format_of_coxloss": format_of_coxloss,
        "missing_mode": getattr(args, "missing_mode", "model_gen"),
        "missing_pattern": getattr(args, "missing_pattern", ""),
        "missing_seed": getattr(args, "missing_seed", getattr(args, "start_seed", 0)),
    })

  
    data_pack_dir = getattr(args, "data_pack_dir", None)
    if data_pack_dir and hgcn_pack_complete(data_pack_dir):
        data_pack_dir = Path(data_pack_dir)
        patients = joblib.load(data_pack_dir / "patients.pkl")
        sur_and_time = joblib.load(data_pack_dir / "sur_and_time.pkl")
        all_data = joblib.load(data_pack_dir / "all_data.pkl")
    else:
        data_root_dir = getattr(args, "data_root_dir", None)
        gene_dir = getattr(args, "gene_dir", None)
        clinic_dir = getattr(args, "clinic_dir", None)
        if not data_root_dir or not gene_dir or not clinic_dir:
            raise ValueError(
                "HGCN reads native modality pkl dirs under hgcn data. "
                "Pass data_root_dir, gene_dir and clinic_dir, "
                "or provide a complete --data_pack_dir."
            )
        data_root_dir = remap_workspace_path_to_hgcn_data(data_root_dir, ROOT_DIR)
        gene_dir = remap_workspace_path_to_hgcn_data(gene_dir, ROOT_DIR)
        clinic_dir = remap_workspace_path_to_hgcn_data(clinic_dir, ROOT_DIR)
        patients, sur_and_time, all_data = load_hgcn_graphs_from_dirs(
            _normalize_study_name(cancer_type),
            data_root_dir=data_root_dir,
            gene_dir=gene_dir,
            clinic_dir=clinic_dir,
            repo_root=ROOT_DIR,
        )

    split_dir = _resolve_split_dir(args, cancer_type)
    patients = [str(case_id) for case_id in patients if str(case_id) in all_data]
    if not patients:
        raise ValueError("HGCN pack has no assembled graphs overlapping patients.pkl")
    missing_labels = [case_id for case_id in patients if case_id not in sur_and_time]
    if missing_labels:
        raise KeyError(f"sur_and_time missing {len(missing_labels)} assembled cases, e.g. {missing_labels[:3]}")
    patient_sur_type, patient_and_time, kf_label = get_patients_information(patients,sur_and_time)
    n_event = int(sum(int(patient_sur_type[case_id]) == 1 for case_id in patients))
    n_censored = len(patients) - n_event
    print(
        f"[HGCN] event encoding: 1=event, 0=censored; n_event={n_event} n_censored={n_censored} n_graphs={len(patients)}",
        flush=True,
    )


    all_seed_patients = []
        
    all_fold_test_ci = []
    all_fold_test_ci_cli_3 = []


    all_all_ci = []
    all_gnn_time = []
    all_each_model_time = []
    all_fold_each_model_ci = []
    seed_summary_rows = []
    all_fold_test_rows = []
    all_eval_subset_rows = []
    ##

    all_epoch_val_loss = []
    all_epoch_test_loss = []
    
    all_epoch_test_img_ci = []
    all_epoch_test_rna_ci = []
    all_epoch_test_cli_ci = []
    
    all_epoch_train_ci = []
    all_epoch_val_ci = []
    all_epoch_test_ci = []


    repeat = -1
    for seed in range(start_seed,start_seed+repeat_num):
        repeat+=1
        setup_seed(0)
        seed_dir = build_seed_dir(results_root, seed)
            
        seed_patients = []
        gnn_feature = {}
        one_test_feature = {}
        gnn_time = {}
        each_model_time = {'img':{},'rna':{},'cli':{},'imgrna':{},'imgcli':{},'rnacli':{}}

        val_gnn_time = {}
        test_fold_ci = []

        val_fold_ci = []
        test_each_model_ci = {'img':[],'rna':[],'cli':[],'imgrna':[],'imgcli':[],'rnacli':[]}
        seed_fold_rows = []
        eval_subset_rows = []

        train_fold_ci=[]
        fold_att_1 = {}
        fold_att_2 = {}
        
        epoch_train_loss = []
        epoch_val_loss = []
        epoch_test_loss = []
        epoch_train_ci = []
        epoch_val_ci = []
        epoch_test_ci = []
    

        n_fold = 0

        kf = StratifiedKFold(n_splits= 5,shuffle=True,random_state = seed)
        for train_index, test_index in kf.split(patients,kf_label):
            fold_patients = []
            n_fold+=1
            print('fold: ',n_fold)
             
            if fusion_model == 'fusion_model_mae_2':
                model = fusion_model_mae_2(img_in_feats=_infer_hgcn_in_feats(all_data, 'x_img', 1024),
                               rna_in_feats=_infer_hgcn_in_feats(all_data, 'x_rna', 1024),
                               cli_in_feats=_infer_hgcn_in_feats(all_data, 'x_cli', 1024),
                               n_hidden=args.n_hidden,
                               out_classes=args.out_classes,
                               dropout=drop_out_ratio,
                               train_type_num = len(args.train_use_type)
                                      ).to(device)

            optimizer=Adam(model.parameters(),lr=lr,weight_decay=5e-4)

            
            if args.if_fit_split:
                split_csv_path = os.path.join(split_dir, f"splits_{n_fold-1}.csv")
                train_data, val_data, test_data = _load_split_csv(split_csv_path)
                train_data = _intersect_split_ids(train_data, patients, "train", split_csv_path)
                val_data = _intersect_split_ids(val_data, patients, "val", split_csv_path)
                test_data = _intersect_split_ids(test_data, patients, "test", split_csv_path)
            else:
                t_train_data = np.array(patients)[train_index]
                t_l = []
                for x in t_train_data:
                    t_l.append(patient_sur_type[x])
                train_data, val_data ,_ , _ = train_test_split(t_train_data,t_train_data,test_size=0.25,random_state=1,stratify=t_l)         
                test_data = np.array(patients)[test_index]

            if getattr(args, "missing_mode", "model_gen") == "unified_mask_csv":
                study_name = getattr(args, "study", None) or _normalize_study_name(cancer_type)
                mask_csv = unified_mask_csv_path(args, n_fold - 1, study=study_name)
                args._unified_mask_lookup = load_fold_mask_lookup(mask_csv)
                print(f"[missing] HGCN unified mask: {mask_csv}")
            else:
                args._unified_mask_lookup = None
            print(len(train_data),len(val_data),len(test_data))
            fold_patients.append(train_data)
            fold_patients.append(val_data)
            fold_patients.append(test_data)
            seed_patients.append(fold_patients)
            fold_dir = build_fold_dir(seed_dir, n_fold)
   
            
            best_loss = 9999
            best_val_ci = 0
            tmp_train_ci=0

            for epoch in range(epochs):
                
                if if_adjust_lr:
                    adjust_learning_rate(optimizer, lr, epoch, lr_step=20, lr_gamma=args.adjust_lr_ratio)
                
                
                
                
                all_loss,t_train_ci,t_train_ci_img,t_train_ci_rna,t_train_ci_cli = train_a_epoch(model,train_data,all_data,patient_and_time,patient_sur_type,batch_size,optimizer,epoch, format_of_coxloss, args)
                
                t_test_loss,test_ci,test_img_ci,test_rna_ci,test_cli_ci = prediction(all_data,model,test_data,patient_and_time,patient_sur_type,args)  
                v_loss,val_ci,val_img_ci,val_rna_ci,val_cli_ci = prediction(all_data,model,val_data,patient_and_time,patient_sur_type,args)
              
                
                
                if val_ci >= best_val_ci and epoch>1 :
                    best_val_ci = val_ci
                    tmp_train_ci = t_train_ci
                    print(val_ci)
                    t_model = copy.deepcopy(model)

                print("epoch：{:2d}，train_loos：{:.4f},train_ci：{:.4f},val_loos：{:.4f},val_ci：{:.4f},test_loos：{:.4f},test_ci：{:.5f}".format(epoch,all_loss,t_train_ci,v_loss,val_ci,t_test_loss,test_ci)) 

    

            t_model.eval() 

            
            t_test_loss,test_ci,_,_,_ = prediction(all_data,t_model,test_data,patient_and_time,patient_sur_type,args)
            

            test_fold_ci.append(test_ci)
            val_fold_ci.append(best_val_ci)
            train_fold_ci.append(tmp_train_ci)

            subset_preds = {name: {} for name in ALL_EVAL_SUBSETS}
            fold_fusion_test_ci = {}
            requested_subsets = _requested_eval_subsets(args)
            with torch.no_grad():
                for id in test_data:  
                    data = all_data[id].to(device)
                    (one_x,multi_x),fea,(att_1,att_2),_ = t_model(data,args.train_use_type,args.train_use_type,mix=args.mix)
                    gnn_time[id] = one_x.cpu().detach().numpy()[0]
                    fold_fusion_test_ci[id] = one_x.cpu().detach().numpy()[0]
                    subset_preds["PCG"][id] = one_x.cpu().detach().numpy()[0]
                    print(data.sur_type.cpu().detach().numpy()[0],one_x.cpu().detach().numpy()[0],patient_and_time[id])
                    one_test_feature[id] = {}
                    for subset in requested_subsets:
                        if subset == "PCG":
                            continue
                        use_type = hgcn_use_type_for_subset(subset)
                        (one_,two_),one_fea,(_,_),_ = t_model(data,args.train_use_type,use_type=use_type,mix=args.mix)
                        pred = one_.cpu().detach().numpy()[0]
                        subset_preds[subset][id] = pred
                        if len(use_type) == 1:
                            each_model_time[use_type[0]][id] = pred
                        else:
                            each_model_time[''.join(use_type)][id] = pred
                    del data        
            for subset in requested_subsets or ("PCG",):
                preds = fold_fusion_test_ci if subset == "PCG" else subset_preds[subset]
                t_ci = get_val_ci(preds,patient_and_time,patient_sur_type)
                if subset != "PCG":
                    legacy = ''.join(hgcn_use_type_for_subset(subset))
                    if legacy in test_each_model_ci:
                        test_each_model_ci[legacy].append(t_ci)
                print(len(preds),' ',subset,' ci:',t_ci)
                if requested_subsets:
                    _append_hgcn_subset_row(eval_subset_rows, n_fold - 1, subset, t_ci)
                
            test_ci = get_val_ci(fold_fusion_test_ci,patient_and_time,patient_sur_type)
            print('all ci:',test_ci)
            all_fold_test_rows.append(_hgcn_fold_metric_row(test_ci))


            torch.save(t_model.state_dict(), str(fold_dir / f"{label}_{seed}_{n_fold}.pth"))
            seed_fold_rows.append({
                "seed": seed,
                "fold": n_fold,
                "val_cindex": best_val_ci,
                "test_cindex": test_ci,
                "train_cindex": tmp_train_ci,
                "all_cindex": test_ci,
            })
            del model, train_data, test_data, t_model
            

        print('seed: ',seed)
        print('test fold ci:')
        for x in test_fold_ci:
            print(x)
          
        print('all ci:')
        print(get_all_ci(gnn_time,patient_and_time,patient_sur_type))
        
        print('val fold ci:')
        for x in val_fold_ci:
            print(x)

    
        all_fold_test_ci.append(test_fold_ci) 
        all_fold_each_model_ci.append(test_each_model_ci)
        all_all_ci.append(get_all_ci(gnn_time,patient_and_time,patient_sur_type))
        all_gnn_time.append(gnn_time)
        all_each_model_time.append(each_model_time)
        all_eval_subset_rows.extend(eval_subset_rows)
        pd.DataFrame(seed_fold_rows).to_csv(seed_dir / "fold_result.csv", index=False)
        joblib.dump(gnn_time, seed_dir / "all_gnn_time.pkl")
        joblib.dump(each_model_time, seed_dir / "all_each_model_time.pkl")
        seed_summary_rows.append({
            "seed": seed,
            "mean_test_cindex": float(np.mean(test_fold_ci)) if test_fold_ci else np.nan,
            "mean_val_cindex": float(np.mean(val_fold_ci)) if val_fold_ci else np.nan,
            "all_cindex": float(get_all_ci(gnn_time, patient_and_time, patient_sur_type)) if gnn_time else np.nan,
        })

    
    print('summary :')
    print(label)  
    
    print('fusion test fold ci')
    for i,x in enumerate(all_fold_test_ci):       
        print(x)
        
        
    for i,type_name in enumerate(['img','rna','cli','imgrna','imgcli','rnacli']): 

        print(type_name,' ci:')
        for fold_ in all_fold_each_model_ci:
            print(fold_[type_name])


    pd.DataFrame(seed_summary_rows).to_csv(results_root / "seed_summary.csv", index=False)
    _write_hgcn_fold_test_result(results_root, all_fold_test_rows)
    joblib.dump(all_gnn_time, results_root / "all_gnn_time.pkl")
    joblib.dump(all_each_model_time, results_root / "all_each_model_time.pkl")
    if _should_write_eval_subsets(args):
        write_eval_subset_outputs(results_root, all_eval_subset_rows)
    
def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer_type", type=str, default="lihc", help="Cancer type")
    parser.add_argument("--img_cox_loss_factor", type=float, default=5, help="img_cox_loss_factor")
    parser.add_argument("--rna_cox_loss_factor", type=float, default=1, help="rna_cox_loss_factor")
    parser.add_argument("--cli_cox_loss_factor", type=float, default=5, help="cli_cox_loss_factor")
    parser.add_argument("--train_use_type", type=list, default=['img','rna','cli'], help='train_use_type,Please keep the relative order of img, rna, cli')
    parser.add_argument("--format_of_coxloss", type=str, default="multi", help="format_of_coxloss:multi,one")
    parser.add_argument("--add_mse_loss_of_mae", action='store_true', default=True, help="add_mse_loss_of_mae")
    parser.add_argument("--mse_loss_of_mae_factor", type=float, default=5, help="mae_loss_factor")
    parser.add_argument("--start_seed", type=int, default=0, help="start_seed")
    parser.add_argument("--repeat_num", type=int, default=5, help="Number of repetitions of the experiment")
    parser.add_argument("--fusion_model", type=str, default="fusion_model_mae_2", help="")
    parser.add_argument("--drop_out_ratio", type=float, default=0.5, help="Drop_out_ratio")
    parser.add_argument("--lr", type=float, default=0.00003, help="Learning rate of model training")
    parser.add_argument("--epochs", type=int, default=60, help="Cycle times of model training")
    parser.add_argument("--batch_size", type=int, default=32, help="Data volume of model training once")
    parser.add_argument("--n_hidden", type=int, default=512, help="Model middle dimension")    
    parser.add_argument("--out_classes", type=int, default=512, help="Model out dimension")
    parser.add_argument("--mix", action='store_true', default=True, help="mix mae")
    parser.add_argument("--if_adjust_lr", action='store_true', default=True, help="if_adjust_lr")
    parser.add_argument("--adjust_lr_ratio", type=float, default=0.5, help="adjust_lr_ratio")
    parser.add_argument("--if_fit_split", action='store_true', default=False, help="fixed division/random division")
    parser.add_argument("--split_root", type=str, default="./splits/5foldcv", help="root directory for split csv files")
    parser.add_argument("--split_dir", type=str, default=None, help="override split directory for one cohort")
    parser.add_argument("--data_pack_dir", type=str, default=None, help="directory containing patients.pkl / sur_and_time.pkl / all_data.pkl")
    parser.add_argument("--data_root_dir", type=str, default=None, help="HGCN WSI pkl directory, usually hgcn data/<study>/P/kimianet")
    parser.add_argument("--clinic_dir", type=str, default=None, help="HGCN clinic pkl directory, usually hgcn data/<study>/C/L{k}")
    parser.add_argument("--gene_dir", type=str, default=None, help="HGCN gene pkl directory, usually hgcn data/<study>/G/msigdb_gsea_families")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--exp_group", type=str, default="HGCN")
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--details", type=str, default='', help="Experimental details")
    parser.add_argument("--missing_mode", type=str, default="model_gen", choices=["model_gen", "unified_mask_csv"])
    parser.add_argument("--missing_pattern", type=str, default="")
    parser.add_argument("--missing_seed", type=int, default=None)
    parser.add_argument("--eval_modalities", type=str, default="off")
    parser.add_argument("--study", type=str, default=None)
    parser.add_argument("--k", type=int, default=5)
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        args=get_params()
        main(args)
    except Exception as exception:
        raise
    
       

 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
