from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any


MODALITY_KEYS = ("wsi", "gene", "clinic")
POE_TRAIN_MODALITIES = {
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
}
MISSING_MODES = ("model_gen", "unified_mask_csv")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise ValueError(f"Cannot parse availability value {value!r}.")


def _normalize_case_id(case_id: str) -> str:
    return str(case_id).strip().upper()[:12]


def _stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def parse_missing_pattern(pattern: str | None) -> tuple[int, int, int]:
    raw = str(pattern or "").strip().lower()
    if not raw:
        raise ValueError("unified_mask_csv requires --missing_pattern, e.g. 60,0,0")
    raw = raw.replace("p", "").replace("-", "_").replace("/", "_").replace(" ", "")
    tokens = raw.split(",") if "," in raw else raw.split("_")
    tokens = [token for token in tokens if token]
    if len(tokens) != 3:
        raise ValueError(f"Invalid missing_pattern {pattern!r}. Expected wsi,gene,clinic percents.")
    rates: list[int] = []
    for token in tokens:
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid missing_pattern {pattern!r}.") from exc
        if value < 0 or value > 100:
            raise ValueError(f"Missing percent out of range in {pattern!r}.")
        rates.append(value)
    return rates[0], rates[1], rates[2]


def pattern_tag(rates: tuple[int, int, int]) -> str:
    return f"p{rates[0]}_{rates[1]}_{rates[2]}"


def format_missing_pattern(rates: tuple[int, int, int]) -> str:
    return f"{rates[0]},{rates[1]},{rates[2]}"


def default_mask_root(args) -> Path:
    results_dir = Path(getattr(args, "results_dir", "./results"))
    exp_group = str(getattr(args, "exp_group", "default"))
    parts = list(results_dir.parts)
    if exp_group in parts:
        idx = parts.index(exp_group)
        return Path(*parts[: idx + 1]) / "masks"
    return results_dir / exp_group / "masks"


def unified_mask_dir(args, study: str | None = None) -> Path:
    rates = parse_missing_pattern(getattr(args, "missing_pattern", ""))
    study_name = study or args.study
    seed = int(getattr(args, "missing_seed", getattr(args, "seed", 1)))
    return default_mask_root(args) / study_name / pattern_tag(rates) / f"seed_{seed}"


def unified_mask_csv_path(args, fold: int, study: str | None = None) -> Path:
    return unified_mask_dir(args, study=study) / f"fold_{int(fold)}.csv"


def all_true_avail() -> dict[str, bool]:
    return {name: True for name in MODALITY_KEYS}


def _read_split_case_ids(split_csv_path: Path) -> dict[str, list[str]]:
    if not split_csv_path.exists():
        raise FileNotFoundError(f"Split csv not found: {split_csv_path}")
    with split_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{split_csv_path} has no header.")
        fieldnames = [str(name) for name in reader.fieldnames]
        missing = [name for name in ("train", "val", "test") if name not in fieldnames]
        if missing:
            raise ValueError(f"{split_csv_path} missing columns: {missing}")
        buckets = {name: [] for name in ("train", "val", "test")}
        seen = {name: set() for name in buckets}
        for row in reader:
            for split_name in buckets:
                case_id = str(row.get(split_name, "")).strip()
                if not case_id:
                    continue
                normalized = _normalize_case_id(case_id)
                if normalized in seen[split_name]:
                    continue
                seen[split_name].add(normalized)
                buckets[split_name].append(normalized)
    return buckets


def _sample_split_masks(
    case_ids: list[str],
    rates: tuple[int, int, int],
    rng: random.Random,
) -> dict[str, dict[str, bool]]:
    avail = {case_id: all_true_avail() for case_id in case_ids}
    n = len(case_ids)
    if n == 0:
        return avail
    for name, percent in zip(MODALITY_KEYS, rates):
        k = int(round(percent / 100.0 * n))
        k = min(max(k, 0), n)
        if k == 0:
            continue
        for case_id in rng.sample(case_ids, k):
            avail[case_id][name] = False
    for case_id in case_ids:
        if not any(avail[case_id].values()):
            restore = MODALITY_KEYS[rng.randrange(len(MODALITY_KEYS))]
            avail[case_id][restore] = True
    return avail


def _missing_rate(rows: list[dict[str, Any]], split_name: str) -> dict[str, float]:
    split_rows = [row for row in rows if row["split"] == split_name]
    n = len(split_rows)
    if n == 0:
        return {name: 0.0 for name in MODALITY_KEYS}
    return {
        name: 1.0 - (sum(int(row[name]) for row in split_rows) / n)
        for name in MODALITY_KEYS
    }


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "split", *MODALITY_KEYS])
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def _acquire_lock(lock_path: Path, timeout_s: float = 300.0) -> None:
    start = time.time()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return
        except FileExistsError:
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Timed out waiting for mask lock: {lock_path}")
            time.sleep(0.2)


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def load_fold_mask_lookup(path: Path) -> dict[str, dict[str, bool]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing unified mask csv: {path}")
    lookup: dict[str, dict[str, bool]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "split", *MODALITY_KEYS}
        if reader.fieldnames is None or required.difference(reader.fieldnames):
            raise ValueError(f"{path} is missing required mask columns {sorted(required)}")
        for row in reader:
            case_id = _normalize_case_id(row["case_id"])
            lookup[case_id] = {name: _as_bool(row[name]) for name in MODALITY_KEYS}
            if not any(lookup[case_id].values()):
                raise ValueError(f"{path} has an all-missing row for {case_id!r}.")
    if not lookup:
        raise ValueError(f"{path} is empty.")
    return lookup


def generate_unified_fold_mask(args, fold: int, study: str | None = None) -> Path:
    study_name = study or args.study
    split_dir = Path(getattr(args, "split_dir", ""))
    if study is not None and study != getattr(args, "study", study):
        current = Path(getattr(args, "split_dir", f"splits/{getattr(args, 'which_splits', '5foldcv')}/{args.study}"))
        split_dir = current.parent / study_name
    split_csv = split_dir / f"splits_{int(fold)}.csv"
    out_csv = unified_mask_csv_path(args, fold, study=study_name)
    rates = parse_missing_pattern(getattr(args, "missing_pattern", ""))
    seed = int(getattr(args, "missing_seed", getattr(args, "seed", 1)))
    lock_path = out_csv.with_suffix(".csv.lock")
    _acquire_lock(lock_path)
    try:
        if out_csv.exists():
            load_fold_mask_lookup(out_csv)
            return out_csv
        split_ids = _read_split_case_ids(split_csv)
        rows: list[dict[str, Any]] = []
        for split_name, case_ids in split_ids.items():
            rng = random.Random(_stable_seed(seed, study_name, format_missing_pattern(rates), fold, split_name))
            sampled = _sample_split_masks(case_ids, rates, rng)
            for case_id in case_ids:
                rows.append({
                    "case_id": case_id,
                    "split": split_name,
                    **{name: int(sampled[case_id][name]) for name in MODALITY_KEYS},
                })
        _write_csv_atomic(out_csv, rows)
        manifest_path = out_csv.parent / "manifest.json"
        manifest = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = {}
        fold_stats = {
            "csv": str(out_csv),
            "n": {split_name: len(ids) for split_name, ids in split_ids.items()},
            "actual_missing_rate": {
                split_name: _missing_rate(rows, split_name)
                for split_name in ("train", "val", "test")
            },
        }
        manifest.update({
            "exp_group": getattr(args, "exp_group", "default"),
            "study": study_name,
            "pattern": format_missing_pattern(rates),
            "seed": seed,
            "source_split_dir": str(split_dir),
        })
        folds = dict(manifest.get("folds", {}))
        folds[str(int(fold))] = fold_stats
        manifest["folds"] = folds
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return out_csv
    finally:
        _release_lock(lock_path)


def _studies_for_masks(args) -> list[str]:
    studies = [args.study]
    raw_value = getattr(args, "poe_stage1_studies", "")
    if getattr(args, "modality", "") == "survtri_poe_vae_b_crossstage1" and raw_value:
        for token in str(raw_value).split(","):
            study = token.strip()
            if not study:
                continue
            if not study.startswith("tcga_"):
                study = f"tcga_{study}"
            if study not in studies:
                studies.append(study)
    return studies


def _folds_for_study(args, study: str) -> list[int]:
    split_dir = Path(getattr(args, "split_dir", f"splits/{getattr(args, 'which_splits', '5foldcv')}/{args.study}"))
    if study != getattr(args, "study", study):
        split_dir = split_dir.parent / study
    folds = sorted(
        int(path.stem.split("_")[-1])
        for path in split_dir.glob("splits_*.csv")
        if path.stem.split("_")[-1].isdigit()
    )
    if folds:
        return folds
    n_folds = int(getattr(args, "k", 5) or 5)
    return list(range(n_folds))


def ensure_exp_group_masks(args) -> None:
    if getattr(args, "missing_mode", "model_gen") != "unified_mask_csv":
        return
    rates = parse_missing_pattern(getattr(args, "missing_pattern", ""))
    args.missing_pattern = format_missing_pattern(rates)
    mask_root = default_mask_root(args)
    mask_root.mkdir(parents=True, exist_ok=True)
    generated = []
    for study in _studies_for_masks(args):
        for fold in _folds_for_study(args, study):
            generated.append(generate_unified_fold_mask(args, fold, study=study))
    print(
        "[missing] mode=unified_mask_csv "
        f"pattern={args.missing_pattern} "
        f"seed={getattr(args, 'missing_seed', args.seed)} "
        f"dir={mask_root} "
        f"files={len(generated)}"
    )


def prepare_missing_protocol(args):
    mode = getattr(args, "missing_mode", "model_gen")
    if mode not in MISSING_MODES:
        raise ValueError(f"Unsupported missing_mode {mode!r}.")
    if getattr(args, "missing_seed", None) is None:
        args.missing_seed = getattr(args, "seed", 1)
    args.missing_seed = int(args.missing_seed)
    if mode == "unified_mask_csv":
        requested_dropout = float(getattr(args, "poe_modality_dropout", 0.0) or 0.0)
        if requested_dropout != 0.0:
            print(
                "[missing] unified_mask_csv forces poe_modality_dropout 0 "
                f"(was {requested_dropout})"
            )
        args.poe_modality_dropout = 0.0
        ensure_exp_group_masks(args)
    else:
        print(
            "[missing] mode=model_gen "
            f"poe_modality_dropout={getattr(args, 'poe_modality_dropout', 0.2)} "
            f"seed={args.missing_seed}"
        )
    return args


def model_gen_case_avail(
    case_id: str,
    *,
    fold: int,
    missing_seed: int,
    drop_prob: float,
) -> dict[str, bool]:
    if drop_prob <= 0:
        return all_true_avail()
    rng = random.Random(_stable_seed(missing_seed, int(fold), _normalize_case_id(case_id), f"{drop_prob:.6f}"))
    avail = {name: rng.random() >= drop_prob for name in MODALITY_KEYS}
    if not any(avail.values()):
        avail[MODALITY_KEYS[rng.randrange(len(MODALITY_KEYS))]] = True
    return avail


def resolve_case_availability(
    case_id: str,
    *,
    missing_mode: str = "model_gen",
    mask_lookup: dict[str, dict[str, bool]] | None = None,
    mask_csv_path: str | Path | None = None,
    is_training: bool = False,
    modality: str | None = None,
    fold: int = 0,
    missing_seed: int = 1,
    drop_prob: float = 0.0,
) -> dict[str, bool]:
    normalized = _normalize_case_id(case_id)
    if missing_mode == "unified_mask_csv":
        lookup = mask_lookup
        if lookup is None:
            if mask_csv_path is None:
                raise ValueError("unified_mask_csv requires a loaded mask lookup or csv path.")
            lookup = load_fold_mask_lookup(Path(mask_csv_path))
        try:
            return dict(lookup[normalized])
        except KeyError as exc:
            raise KeyError(
                f"Case {normalized!r} is missing from unified mask csv {mask_csv_path!r}."
            ) from exc
    if is_training and modality in POE_TRAIN_MODALITIES and drop_prob > 0:
        return model_gen_case_avail(
            normalized,
            fold=fold,
            missing_seed=missing_seed,
            drop_prob=drop_prob,
        )
    return all_true_avail()


def avail_to_hgcn_in_mask(avail: dict[str, bool], train_use_type: list[str]):
    import numpy as np

    key_map = {"img": "wsi", "rna": "gene", "cli": "clinic"}
    flags = []
    for name in train_use_type:
        avail_key = key_map.get(name)
        if avail_key is None:
            raise KeyError(f"Unsupported HGCN modality {name!r}.")
        if avail_key not in avail:
            raise KeyError(f"Missing availability key {avail_key!r} for HGCN.")
        flags.append(not bool(avail[avail_key]))
    return np.array(flags, dtype=bool).reshape(1, 1, -1)
