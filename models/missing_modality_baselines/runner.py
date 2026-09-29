from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace


ROOT_DIR = Path(__file__).resolve().parents[2]


@contextmanager
def _prepend_path(path: Path):
    path_str = str(path)
    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        if sys.path and sys.path[0] == path_str:
            sys.path.pop(0)
        elif path_str in sys.path:
            sys.path.remove(path_str)


def _load_module(module_name: str, file_path: Path, extra_path: Path):
    with _prepend_path(extra_path):
        if str(ROOT_DIR) not in sys.path:
            sys.path.append(str(ROOT_DIR))
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module



def _load_local(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hgcn_paths = _load_local(
    "survpgc_hgcn_paths",
    Path(__file__).resolve().parent / "hgcn_paths.py",
)


def run_hgcn_from_args(args):
    hgcn_dir = ROOT_DIR / "models" / "missing_modality_baselines" / "third_party" / "HGCN" / "HGCN_code"
    module = _load_module("survpgc_hgcn_train", hgcn_dir / "train.py", hgcn_dir)
    cancer_type = args.study.replace("tcga_", "")
    remapped = hgcn_paths.remap_hgcn_feature_dirs(
        data_root_dir=args.data_root_dir,
        clinic_dir=args.clinic_dir,
        gene_dir=args.gene_dir,
        repo_root=ROOT_DIR,
    )
    hgcn_args = SimpleNamespace(
        cancer_type=cancer_type,
        img_cox_loss_factor=5,
        rna_cox_loss_factor=1,
        cli_cox_loss_factor=5,
        train_use_type=['img', 'rna', 'cli'],
        format_of_coxloss='multi',
        add_mse_loss_of_mae=True,
        mse_loss_of_mae_factor=5,
        start_seed=getattr(args, "seed", 0),
        repeat_num=1,
        fusion_model='fusion_model_mae_2',
        drop_out_ratio=0.5,
        lr=getattr(args, "lr", 3e-5),
        epochs=getattr(args, "max_epochs", 60),
        batch_size=getattr(args, "batch_size", 32),
        n_hidden=512,
        out_classes=512,
        mix=True,
        if_adjust_lr=True,
        adjust_lr_ratio=0.5,
        if_fit_split=True,
        split_root=getattr(args, "split_root", str(ROOT_DIR / "splits" / "5foldcv")),
        split_dir=getattr(args, "split_dir", None),
        data_root_dir=str(remapped["data_root_dir"]),
        clinic_dir=str(remapped["clinic_dir"]),
        gene_dir=str(remapped["gene_dir"]),
        data_pack_dir=getattr(args, "data_pack_dir", None),
        results_dir=getattr(args, "results_dir", "./results"),
        exp_group=getattr(args, "exp_group", "HGCN"),
        run_name=getattr(args, "run_name", f"{args.study}__hgcn"),
        details=getattr(args, "details", ""),
        missing_mode=getattr(args, "missing_mode", "model_gen"),
        missing_pattern=getattr(args, "missing_pattern", ""),
        missing_seed=getattr(args, "missing_seed", getattr(args, "seed", 1)),
        k=getattr(args, "k", 5),
        seed=getattr(args, "seed", 1),
        study=getattr(args, "study", None),
        split_csv=getattr(args, "split_csv", None),
        which_splits=getattr(args, "which_splits", "5foldcv"),
        eval_modalities=getattr(args, "eval_modalities", "off"),
    )
    return module.main(hgcn_args)
