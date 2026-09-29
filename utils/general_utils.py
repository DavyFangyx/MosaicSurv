#----> general imports
import os
import pandas as pd 

import torch
import numpy as np
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _format_poe_stage1_studies(raw_value):
    studies = []
    seen = set()
    for token in str(raw_value).split(","):
        study = token.strip()
        if not study:
            continue
        if not study.startswith("tcga_"):
            study = f"tcga_{study}"
        if study not in seen:
            studies.append(study)
            seen.add(study)
    return studies

def _infer_wsi_encoding_dim(data_root_dir):
    r"""
    Infer WSI embedding dim from the first available .pt file under data_root_dir.
    This keeps direct `python main.py ... --data_root_dir ...` runs aligned with
    the selected WSI_EXPERIMENT without requiring a manual --encoding_dim override.
    """
    import glob

    pt_files = sorted(glob.glob(os.path.join(data_root_dir, "*.pt")))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found under WSI data_root_dir: {data_root_dir}")

    sample = torch.load(pt_files[0], map_location="cpu")
    if sample.ndim < 2:
        raise ValueError(
            f"Expected WSI embedding tensor with ndim >= 2, got shape {tuple(sample.shape)} from {pt_files[0]}"
        )
    return int(sample.shape[-1])

def _prepare_for_experiment(args):
    r"""
    Creates experiment code which will be used for identifying the experiment later on. Uses the experiment code to make results dir.
    Prints and logs the important settings of the experiment. Loads the pathway composition dataframe and stores in args for future use.

    Args:
        - args : argparse.Namespace
    
    Returns:
        - args : argparse.Namespace

    """

    args.device = device
    print(args.device)
    if not getattr(args, 'split_dir', None):
        args.split_dir = os.path.join("splits", args.which_splits, args.study)
    if getattr(args, 'data_root_dir', None):
        inferred_wsi_dim = _infer_wsi_encoding_dim(args.data_root_dir)
        if getattr(args, 'encoding_dim', None) != inferred_wsi_dim:
            print(f"WSI encoding dim inferred from {args.data_root_dir}: {inferred_wsi_dim} (override {args.encoding_dim})")
            args.encoding_dim = inferred_wsi_dim
    args.combined_study = args.study
    args = _get_custom_exp_code(args)
    _seed_torch(args.seed)

    required_paths = {
        'label_file': args.label_file,
        'clinical_file': getattr(args, 'clinical_file', None),
        'omics_dir': args.omics_dir,
        'data_root_dir': args.data_root_dir,
        'clinic_dir': args.clinic_dir,
        'gene_dir': args.gene_dir,
        'split_dir': args.split_dir,
    }
    missing = [f"{name}={path}" for name, path in required_paths.items() if path and not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing required training inputs:\n" + "\n".join(missing))

    assert os.path.isdir(args.split_dir)
    print('Split dir:', args.split_dir)

    #---> where to stroe the experiment related assets
    _create_results_dir(args)

    #---> store the settings
    settings = {'num_splits': args.k, 
                'k_start': args.k_start,
                'k_end': args.k_end,
                'task': args.task,
                'max_epochs': args.max_epochs, 
                'results_dir': args.results_dir, 
                'lr': args.lr,
                'experiment': args.study,
                'label_file': args.label_file,
                'clinical_file': getattr(args, 'clinical_file', None),
                'omics_dir': args.omics_dir,
                'data_root_dir': args.data_root_dir,
                'clinic_dir': args.clinic_dir,
                'gene_dir': args.gene_dir,
                'reg': args.reg,
                # 'label_frac': args.label_frac,
                'bag_loss': args.bag_loss,
                'seed': args.seed,
                'weighted_sample': args.weighted_sample,
                'opt': args.opt,
                "num_patches":args.num_patches,
                "dropout":args.encoder_dropout,
                "type_of_path":args.type_of_path,
                'split_dir': args.split_dir
                }
    
    #---> bookkeping
    _print_and_log_experiment(args, settings)

    #---> load composition df 
    composition_df = pd.read_csv("./datasets_csv/pathway_compositions/{}_comps.csv".format(args.type_of_path), index_col=0)
    composition_df.sort_index(inplace=True)
    args.composition_df = composition_df

    return args

def _print_and_log_experiment(args, settings):
    r"""
    Prints the expeirmental settings and stores them in a file 
    
    Args:
        - args : argspace.Namespace
        - settings : dict 
    
    Return:
        - None
        
    """
    with open(args.results_dir + '/experiment.txt', 'w') as f:
        print(settings, file=f)

    f.close()

    print("")
    print("################# Settings ###################")
    for key, val in settings.items():
        print("{}:  {}".format(key, val))
    print("")

def _get_custom_exp_code(args):
    r"""
    Updates the argparse.NameSpace with a custom experiment code.

    Args:
        - args (NameSpace)

    Returns:
        - args (NameSpace)

    """
    dataset_path = 'datasets_csv/all_survival_endpoints'
    param_code = ''

    #----> Study 
    param_code += args.study + "_"

    #----> Loss Function
    param_code += '_%s' % args.bag_loss
    param_code += '_a%s' % str(args.alpha_surv)
    
    #----> Learning Rate
    param_code += '_lr%s' % format(args.lr, '.0e')

    #----> Regularization
    # if args.reg_type == 'L1':
    #   param_code += '_%sreg%s' % (args.reg_type, format(args.reg, '.0e'))

    # if args.reg and args.reg_type == "L2":
    param_code += "_l2Weight_{}".format(args.reg)

    param_code += '_%s' % args.which_splits.split("_")[0]

    #----> Batch Size
    param_code += '_b%s' % str(args.batch_size)

    # label col 
    param_code += "_" + args.label_col

    param_code += "_dim1_" + str(args.encoding_dim)
    # param_code += "_dim2_" + str(args.encoding_layer_2_dim)
    
    param_code += "_patches_" + str(args.num_patches)
    # param_code += "_dropout_" + str(args.encoder_dropout)

    param_code += "_wsiDim_" + str(args.wsi_projection_dim)
    param_code += "_epochs_" + str(args.max_epochs)
    param_code += "_fusion_" + str(args.fusion)
    param_code += "_modality_" + str(args.modality)
    param_code += "_selected_" + str(args.selected_modalities).replace(",", "_")
    param_code += "_pathT_" + str(args.type_of_path)
    if args.modality == "survtri_poe_vae_b_crossstage1":
        stage1_tag = "_".join(_format_poe_stage1_studies(getattr(args, "poe_stage1_studies", "")))
        if stage1_tag:
            param_code += "_stage1_" + stage1_tag

    #----> Updating
    args.param_code = param_code
    args.dataset_path = dataset_path

    return args


def _seed_torch(seed=7):
    r"""
    Sets custom seed for torch 

    Args:
        - seed : Int 
    
    Returns:
        - None

    """
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _create_results_dir(args):
    r"""
    Creates a dir to store results for this experiment.

    Directory structure:
        {results_dir}/{exp_group}/{run_name}/{modality}/

    The verbose param_code is no longer part of the path; it is written to
    experiment.txt inside the run dir so it remains discoverable.

    Args:
        - args: argspace.Namespace

    Return:
        - None

    """
    exp_group = getattr(args, 'exp_group', 'default')
    run_name  = getattr(args, 'run_name',  'default')

    # survfusion_separate 追加 fusion_type 后缀；mhsa 还追加 CLIP 权重后缀（实验三 27 组合）
    if args.modality == 'survfusion_separate':
        fusion_type = getattr(args, 'fusion_type', 'mhsa')
        folder = f"{args.modality}_{fusion_type}"
        if fusion_type == 'mhsa':
            w_it = int(getattr(args, 'clip_weight_IT', 1.0))
            w_is = int(getattr(args, 'clip_weight_IS', 1.0))
            w_ts = int(getattr(args, 'clip_weight_TS', 1.0))
            folder += f"_{w_it}_{w_is}_{w_ts}"
    elif args.modality in {
        "survtri_snn_concat",
        "survtri_snn_mhsa",
        "survtri_mlp_concat",
        "survtri_mlp_mhsa",
    } and getattr(args, "selected_modalities", "wsi,gene,clinic") != "wsi,gene,clinic":
        folder = f"{args.modality}__{args.selected_modalities.replace(',', '_')}"
    elif args.modality == "survtri_poe_vae":
        folder = f"{args.modality}__{getattr(args, 'poe_variant', 'A')}"
        if getattr(args, "selected_modalities", "wsi,gene,clinic") != "wsi,gene,clinic":
            folder += f"__{args.selected_modalities.replace(',', '_')}"
    elif args.modality in {"survtri_poe_vae_b_kl", "survtri_poe_vae_b_nopretrain"}:
        folder = args.modality
        if getattr(args, "selected_modalities", "wsi,gene,clinic") != "wsi,gene,clinic":
            folder += f"__{args.selected_modalities.replace(',', '_')}"
    elif args.modality == "survtri_poe_vae_b_crossstage1":
        folder = f"{args.modality}__{getattr(args, 'poe_variant', 'B')}"
        stage1_tag = "_".join(_format_poe_stage1_studies(getattr(args, "poe_stage1_studies", "")))
        if stage1_tag:
            folder += f"__stage1_{stage1_tag}"
        if getattr(args, "selected_modalities", "wsi,gene,clinic") != "wsi,gene,clinic":
            folder += f"__{args.selected_modalities.replace(',', '_')}"
    elif args.modality == "modality_concat":
        concat_wsi = getattr(args, "concat_wsi", "resampler")
        concat_impute = getattr(args, "concat_impute", "zero")
        folder = f"{args.modality}__{concat_wsi}__{concat_impute}"
    else:
        folder = args.modality

    args.results_dir = os.path.join(
        args.results_dir,   # base (e.g. ./results)
        exp_group,          # clinic_test / gene_test / param_tuning / ablation / default
        run_name,           # O_origin / lr_0001 / …
        folder,             # survpgc_f / survfusion_separate_mhsa / …
    )
    os.makedirs(args.results_dir, exist_ok=True)

    # Add a .gitignore at the top-level results/ directory (once)
    gitignore_path = os.path.join(args.results_dir.split(os.sep)[0], ".gitignore")
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, "w") as f:
            f.write("*\n*/\n!.gitignore")

def _get_start_end(args):
    r"""
    Which folds are we training on
    
    Args:
        - args : argspace.Namespace
    
    Return:
       folds : np.array 
    
    """
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end
    folds = np.arange(start, end)
    return folds

def _save_splits(split_datasets, column_keys, filename, boolean_style=False):
    splits = [split_datasets[i].metadata['slide_id'] for i in range(len(split_datasets))]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index = True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val'])

    df.to_csv(filename)
    print()

def _series_intersection(s1, s2):
    r"""
    Return insersection of two sets
    
    Args:
        - s1 : set
        - s2 : set 
    
    Returns:
        - pd.Series
    
    """
    return pd.Series(list(set(s1) & set(s2)))

def _print_network(results_dir, net):
    r"""

    Print the model in terminal and also to a text file for storage 
    
    Args:
        - results_dir : String 
        - net : PyTorch model 
    
    Returns:
        - None 
    
    """
    num_params = 0
    num_params_train = 0

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)

    # print(net)

    # fname = "model_" + results_dir.split("/")[-1] + ".txt"
    fname = "model.txt"
    path = os.path.join(results_dir, fname)
    f = open(path, "w")
    f.write(str(net))
    f.write("\n")
    f.write('Total number of parameters: %d \n' % num_params)
    f.write('Total number of trainable parameters: %d \n' % num_params_train)
    f.close()


def _collate_omics(batch):
    r"""
    Collate function for the unimodal omics models 
    
    Args:
        - batch 
    
    Returns:
        - img : torch.Tensor 
        - omics : torch.Tensor 
        - label : torch.LongTensor 
        - event_time : torch.FloatTensor 
        - c : torch.FloatTensor 
        - clinical_data_list : List
        
    """
  
    img = torch.ones([1,1])
    omics = torch.stack([item[1] for item in batch], dim = 0)
    label = torch.LongTensor([item[2].long() for item in batch])
    event_time = torch.FloatTensor([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])

    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[5])

    return [img, omics, label, event_time, c, clinical_data_list]


def _collate_wsi_omics(batch):
    r"""
    Collate function for the unimodal wsi and multimodal wsi + omics  models 
    
    Args:
        - batch 
    
    Returns:
        - img : torch.Tensor 
        - omics : torch.Tensor 
        - label : torch.LongTensor 
        - event_time : torch.FloatTensor 
        - c : torch.FloatTensor 
        - clinical_data_list : List
        - mask : torch.Tensor
        
    """
  
    img = torch.stack([item[0] for item in batch])
    omics = torch.stack([item[1] for item in batch], dim = 0)
    label = torch.LongTensor([item[2].long() for item in batch])
    event_time = torch.FloatTensor([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])

    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[5])

    mask = torch.stack([item[6] for item in batch], dim=0)

    return [img, omics, label, event_time, c, clinical_data_list, mask]

def _collate_MCAT(batch):
    r"""
    Collate function MCAT (pathways version) model
    
    Args:
        - batch 
    
    Returns:
        - img : torch.Tensor 
        - omic1 : torch.Tensor 
        - omic2 : torch.Tensor 
        - omic3 : torch.Tensor 
        - omic4 : torch.Tensor 
        - omic5 : torch.Tensor 
        - omic6 : torch.Tensor 
        - label : torch.LongTensor 
        - event_time : torch.FloatTensor 
        - c : torch.FloatTensor 
        - clinical_data_list : List
        
    """
    
    img = torch.stack([item[0] for item in batch])

    omic1 = torch.cat([item[1] for item in batch], dim = 0).type(torch.FloatTensor)
    omic2 = torch.cat([item[2] for item in batch], dim = 0).type(torch.FloatTensor)
    omic3 = torch.cat([item[3] for item in batch], dim = 0).type(torch.FloatTensor)
    omic4 = torch.cat([item[4] for item in batch], dim = 0).type(torch.FloatTensor)
    omic5 = torch.cat([item[5] for item in batch], dim = 0).type(torch.FloatTensor)
    omic6 = torch.cat([item[6] for item in batch], dim = 0).type(torch.FloatTensor)


    label = torch.LongTensor([item[7].long() for item in batch])
    event_time = torch.FloatTensor([item[8] for item in batch])
    c = torch.FloatTensor([item[9] for item in batch])

    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[10])

    mask = torch.stack([item[11] for item in batch], dim=0)

    return [img, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c, clinical_data_list, mask]

def _collate_porpoise(batch):
    img = torch.stack([item[0] for item in batch])
    omics = torch.stack([item[1] for item in batch], dim=0)
    label = torch.LongTensor([item[2].long() for item in batch])
    event_time = torch.FloatTensor([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])
    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[5])
    mask = torch.stack([item[6] for item in batch], dim=0)

    return [img, omics, label, event_time, c, clinical_data_list, mask]


def _collate_survpath(batch):
    r"""
    Collate function for survpath
    
    Args:
        - batch 
    
    Returns:
        - img : torch.Tensor 
        - omic_data_list : List
        - label : torch.LongTensor 
        - event_time : torch.FloatTensor 
        - c : torch.FloatTensor 
        - clinical_data_list : List
        - mask : torch.Tensor
        
    """
    
    img = torch.stack([item[0] for item in batch])

    omic_data_list = []
    for item in batch:
        omic_data_list.append(item[1])

    label = torch.LongTensor([item[2].long() for item in batch])
    event_time = torch.FloatTensor([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])

    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[5])

    mask = torch.stack([item[6] for item in batch], dim=0)

    return [img, omic_data_list, label, event_time, c, clinical_data_list, mask]


def _collate_survpgc_f(batch):
    img = torch.stack([item[0] for item in batch])

    omic = torch.stack([item[1] for item in batch])

    clinic = torch.stack([item[2] for item in batch])

    label = torch.LongTensor([item[3].long() for item in batch])
    event_time = torch.FloatTensor([item[4] for item in batch])
    c = torch.FloatTensor([item[5] for item in batch])

    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[6])

    mask = torch.stack([item[7] for item in batch], dim=0)

    return [img, omic, clinic, label, event_time, c, clinical_data_list, mask]


def _collate_survtri_poe_vae(batch):
    img = torch.stack([item[0] for item in batch])
    omic = torch.stack([item[1] for item in batch])
    clinic = torch.stack([item[2] for item in batch])

    label = torch.LongTensor([item[3].long() for item in batch])
    event_time = torch.FloatTensor([item[4] for item in batch])
    c = torch.FloatTensor([item[5] for item in batch])

    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[6])

    mask = torch.stack([item[7] for item in batch], dim=0)
    avail = {
        "wsi": torch.tensor([item[8]["wsi"] for item in batch], dtype=torch.bool),
        "gene": torch.tensor([item[8]["gene"] for item in batch], dtype=torch.bool),
        "clinic": torch.tensor([item[8]["clinic"] for item in batch], dtype=torch.bool),
    }

    return [img, omic, clinic, label, event_time, c, clinical_data_list, mask, avail]

def _collate_survpc_f(batch):
    img = torch.stack([item[0] for item in batch])
    clinic = torch.stack([item[1] for item in batch])

    label = torch.LongTensor([item[2].long() for item in batch])
    event_time = torch.FloatTensor([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])

    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[5])

    mask = torch.stack([item[6] for item in batch], dim=0)

    return [img, clinic, label, event_time, c, clinical_data_list, mask]

def _collate_survgc_f(batch):
    img = torch.ones([1, 1])
    clinic = torch.stack([item[1] for item in batch])
    omic = torch.stack([item[2] for item in batch])
    label = torch.LongTensor([item[3].long() for item in batch])
    event_time = torch.FloatTensor([item[4] for item in batch])
    c = torch.FloatTensor([item[5] for item in batch])
    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[6])

    return [img, clinic, omic, label, event_time, c, clinical_data_list]

def _collate_clinic_f(batch):

    img = torch.ones([1, 1])
    clinic = torch.stack([item[1] for item in batch])
    label = torch.LongTensor([item[2].long() for item in batch])
    event_time = torch.FloatTensor([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])

    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[5])

    return [img, clinic, label, event_time, c, clinical_data_list]


def _make_weights_for_balanced_classes_split(dataset):
    r"""
    Returns the weights for each class. The class will be sampled proportionally.
    
    Args: 
        - dataset : SurvivalDataset
    
    Returns:
        - final_weights : torch.DoubleTensor 
    
    """
    N = float(len(dataset))                                           
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.getlabel(idx)                   
        weight[idx] = weight_per_class[y]   

    final_weights = torch.DoubleTensor(weight)

    return final_weights

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)


def _get_split_loader(args, split_dataset, training = False, testing = False, weighted = False, batch_size=1, disable_cox_batch_override=False):
    r"""
    Take a dataset and make a dataloader from it using a custom collate function. 

    Args:
        - args : argspace.Namespace
        - split_dataset : SurvivalDataset
        - training : Boolean
        - testing : Boolean
        - weighted : Boolean 
        - batch_size : Int 
    
    Returns:
        - loader : Pytorch Dataloader 
    
    """

    kwargs = {'num_workers': 0} if device.type == "cuda" else {}
    
    if args.modality in ["mlp_gene", "snn_gene", "mlp_gene_f", "snn_gene_f"]:
        collate_fn = _collate_omics
    elif args.modality in ['clinic_cox',
                           'mlp_clinic_mean', 'mlp_clinic_flatten',
                           'snn_clinic_mean', 'snn_clinic_flatten']:
        collate_fn = _collate_clinic_f
    elif args.modality in ["abmil_wsi", "mlp_wsi", "transmil_wsi"]:
        collate_fn = _collate_wsi_omics
    elif args.modality in ["mcat"]:
        collate_fn = _collate_MCAT
    elif args.modality in ["porpoise"]:
        collate_fn = _collate_porpoise
    elif args.modality == "survpath":
        collate_fn = _collate_survpath
    elif args.modality == "survpath_f":
        collate_fn = _collate_survpath_f
    elif args.modality in [
        "survpgc_f",
        "survfusion_separate",
        "survfusion_noalign",
        "survfusion_joint",
        "survtri_snn_concat",
        "survtri_snn_mhsa",
        "survtri_mlp_concat",
        "survtri_mlp_mhsa",
    ]:
        collate_fn = _collate_survpgc_f
    elif args.modality in [
        "survtri_poe_vae",
        "survtri_poe_vae_b_kl",
        "survtri_poe_vae_b_crossstage1",
        "survtri_poe_vae_b_nopretrain",
        "survtri_poe_vae_b_single",
        "survtri_poe_vae_b_multi",
        "mosaic_surv_twostage",
        "mosaic_surv_frozen",
        "mosaic_surv_single",
        "mosaic_surv_single_enum",
        "mosaic_surv_multi",
        "mosaic_surv",
        "mosaic_surv_kl",
        "mosaic_surv_noenum",
        "mosaic_surv_detached",
        "mosaic_surv_nojeffreys",
        "modality_concat",
        "mvae_poe",
        "mopoe",
    ]:
        collate_fn = _collate_survtri_poe_vae
    elif args.modality in ["survpc", "survpc_f", "mlppc_concat"]:
        collate_fn = _collate_survpc_f
    elif args.modality == "survgc_f":
        collate_fn = _collate_survgc_f
    else:
        raise NotImplementedError

    effective_batch_size = batch_size
    if args.bag_loss == 'cox_surv' and not disable_cox_batch_override:
        effective_batch_size = max(1, len(split_dataset))

    if not testing:
        if training:
            if weighted:
                weights = _make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=effective_batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_fn, drop_last=False, **kwargs)	
            else:
                loader = DataLoader(split_dataset, batch_size=effective_batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate_fn, drop_last=False, **kwargs)
        else:
            loader = DataLoader(split_dataset, batch_size=effective_batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate_fn, drop_last=False, **kwargs)

    else:
        ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
        loader = DataLoader(split_dataset, batch_size=effective_batch_size, sampler = SubsetSequentialSampler(ids), collate_fn = collate_fn, drop_last=False, **kwargs )

    return loader
