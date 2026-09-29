from __future__ import annotations

import csv
from pathlib import Path

from common import (
    CFILM_MODEL,
    STUDIES,
    STUDY_LABELS,
    default_table1_summary_path,
    load_cindex_stats,
    mean_std,
    number,
    parse_io_args,
    read_hparams,
    table_enabled,
    with_hparams,
    write_csv,
)


DEFAULT_THRESHOLDS = {
    "tcga_kirp": 0.8115,
    "tcga_coad": 0.6611,
    "tcga_kirc": 0.6895,
    "tcga_brca": 0.6588,
    "tcga_lihc": 0.7040,
}


def resolve_result(results_root: Path, run_id: str, study: str) -> Path | None:
    canonical = results_root / "Cfilm_Hparam_Eval" / "Table1" / run_id
    if canonical.is_dir():
        matches = sorted(canonical.glob(f"{study}__*/{CFILM_MODEL}/test_result.csv"))
        if matches:
            return matches[0]

    study_tag = study.replace("tcga_", "").upper()
    run_name = (
        f"{study}__L0__cell_norm__uni_v1__"
        f"survtri_poe_vae_C_film__{run_id}"
    )
    legacy = (
        results_root
        / "Cfilm_Hparam_Eval"
        / "Table1_Cindex"
        / f"L0_{study_tag}_poe_model_val"
        / run_name
        / CFILM_MODEL
        / "test_result.csv"
    )
    return legacy if legacy.is_file() else None


def _parse_mean(value: str) -> float | None:
    text = str(value).strip()
    if not text or text == "-":
        return None
    text = text.split("±", 1)[0].strip()
    text = text.split("(", 1)[0].strip()
    try:
        return float(text)
    except ValueError:
        return None


def write_classic_table1_ranking(
    output_root: Path,
    hparams_rows: list[dict[str, str]],
    cfilm_rows: list[dict[str, object]],
    summary_path: Path,
) -> None:
    """Add each Cfilm parameter group to the classic Table1 ranking."""
    with summary_path.open(newline="", encoding="utf-8") as handle:
        classic_rows = list(csv.DictReader(handle))

    output_rows: list[dict[str, object]] = []
    study_values: dict[str, list[tuple[float, str]]] = {study: [] for study in STUDIES}
    for row in classic_rows:
        for study in STUDIES:
            value = _parse_mean(row.get(study, ""))
            if value is not None:
                study_values[study].append((value, str(row.get("model", ""))))

    cfilm_by_run_study = {
        (str(row["run_id"]), str(row["study"])): _parse_mean(str(row["cindex_mean"]))
        for row in cfilm_rows
    }
    for hparams in hparams_rows:
        run_id = hparams["run_id"].strip()
        values: dict[str, float | None] = {}
        ranks: dict[str, int | str] = {}
        for study in STUDIES:
            value = cfilm_by_run_study.get((run_id, study))
            values[study] = value
            candidates = list(study_values[study])
            if value is not None:
                candidates.append((value, f"Cfilm[{run_id}]"))
                candidates.sort(key=lambda item: (-item[0], item[1]))
                ranks[study] = next(
                    rank for rank, (_, name) in enumerate(candidates, start=1)
                    if name == f"Cfilm[{run_id}]"
                )
            else:
                ranks[study] = ""

        valid = [value for value in values.values() if value is not None]
        output_rows.append(
            {
                **hparams,
                **{f"{study}_cindex": f"{values[study]:.4f}" if values[study] is not None else "" for study in STUDIES},
                **{f"{study}_rank": ranks[study] for study in STUDIES},
                "mean_cindex": f"{sum(valid) / len(valid):.4f}" if valid else "",
                "completed_studies": len(valid),
                "grade": (
                    "A"
                    if all(ranks[study] == 1 for study in STUDIES)
                    else (
                        "B"
                        if all(1 <= ranks[study] <= 3 for study in STUDIES)
                        else "C"
                    )
                ),
            }
        )

    mean_candidates: list[tuple[float, str]] = []
    for row in classic_rows:
        value = _parse_mean(row.get("mean", ""))
        if value is not None:
            mean_candidates.append((value, str(row.get("model", ""))))
    for row in output_rows:
        value = _parse_mean(str(row["mean_cindex"]))
        if value is not None:
            mean_candidates.append((value, f"Cfilm[{row['run_id']}]"))
    mean_candidates.sort(key=lambda item: (-item[0], item[1]))
    mean_rank = {name: rank for rank, (_, name) in enumerate(mean_candidates, start=1)}
    for row in output_rows:
        row["mean_rank"] = mean_rank.get(f"Cfilm[{row['run_id']}]", "")

    hparam_fields = list(hparams_rows[0])
    fields = hparam_fields + [
        *[f"{study}_cindex" for study in STUDIES],
        *[f"{study}_rank" for study in STUDIES],
        "mean_cindex", "mean_rank", "completed_studies",
        "grade",
    ]
    write_csv(output_root / "Table1" / "cfilm_vs_classic_ranking.csv", output_rows, fields)

    checklist_rows = [
        {column: row.get(column, "") for column in hparam_fields}
        for row in output_rows
        if row["grade"] in {"A", "B"}
    ]
    write_csv(
        output_root / "Table1" / "cfilm_table1_hparams_rank_le_3.csv",
        checklist_rows,
        hparam_fields,
    )


def collect(
    results_root: Path,
    output_root: Path,
    hparams_path: Path,
    thresholds: dict[str, float] | None = None,
    classic_summary_path: Path | None = None,
) -> list[dict[str, object]]:
    all_hparams_rows = read_hparams(hparams_path)
    hparams_rows = [row for row in all_hparams_rows if table_enabled(row, "T1")]
    thresholds = thresholds or DEFAULT_THRESHOLDS
    aggregate: list[dict[str, object]] = []
    all_detail_rows: list[dict[str, object]] = []

    for hparams in hparams_rows:
        run_id = hparams["run_id"].strip()
        detail_rows: list[dict[str, object]] = []
        means: list[float] = []
        margins: list[float] = []
        passed = 0

        for study in STUDIES:
            source = resolve_result(results_root, run_id, study)
            stats = load_cindex_stats(source) if source is not None else None
            threshold = thresholds[study]
            margin = stats.mean - threshold if stats is not None else None
            is_pass = stats is not None and stats.mean >= threshold
            if stats is not None:
                means.append(stats.mean)
                margins.append(margin)
            if is_pass:
                passed += 1
            detail_rows.append(
                {
                    "run_id": run_id,
                    "study": study,
                    "study_label": STUDY_LABELS[study],
                    "cindex_mean": number(stats.mean if stats else None),
                    "cindex_std": number(stats.std if stats else None),
                    "folds": stats.count if stats else 0,
                    "threshold": number(threshold, 4),
                    "margin": number(margin),
                    "passed": "yes" if is_pass else ("no" if stats else ""),
                    "status": "complete" if stats else "missing",
                    "source": str(source) if source else "",
                }
            )
        all_detail_rows.extend(detail_rows)

        write_csv(
            output_root / "Table1" / run_id / "summary.csv",
            detail_rows,
            [
                "run_id", "study", "study_label", "cindex_mean", "cindex_std",
                "folds", "threshold", "margin", "passed", "status", "source",
            ],
        )
        across = mean_std(means)
        complete = len(means) == len(STUDIES)
        aggregate.append(
            with_hparams(
                {
                    "completed_studies": len(means),
                    "expected_studies": len(STUDIES),
                    "passed_studies": passed,
                    "all_pass": "yes" if complete and passed == len(STUDIES) else "no",
                    "mean_cindex": across.mean if across else None,
                    "across_study_std": across.std if across else None,
                    "min_margin": min(margins) if margins else None,
                    "status": "complete" if complete else "incomplete",
                },
                hparams,
            )
        )

    write_classic_table1_ranking(
        output_root,
        hparams_rows,
        all_detail_rows,
        classic_summary_path or default_table1_summary_path(),
    )
    return aggregate


def main() -> None:
    args = parse_io_args("Collect Table1 Cfilm hyperparameter results")
    collect(args.results_root, args.output_root, args.hparams, classic_summary_path=args.classic_summary)
    print(f"Collected Table1 results into {args.output_root / 'Table1'}")


if __name__ == "__main__":
    main()
