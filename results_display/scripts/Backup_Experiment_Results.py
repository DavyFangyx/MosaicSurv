"""实验结果备份脚本：把当前实验结果快照归档到 results/backup/<日期>_<名称>/。

归档结构（与 results/backup/20261007_rerun_failed_nodes 同形）：

    results/backup/<YYYYMMDD>_<name>/
        raw/            # results/ 下的实验树（默认复制，--move 则 mv 让位）
        display/        # results_display/ 下的显示产物（始终复制）
        list.md         # 归档清单：时间、来源、复制方式、体积

用法（项目根目录）：

# 默认归档官方四大实验的 display 目录（体积小，始终安全）
python results_display/scripts/Backup_Experiment_Results.py --name paper_v1

# 指定要归档的路径（results/... 或 results_display/...，可多个）
python results_display/scripts/Backup_Experiment_Results.py --name rerun \
    --include results/Table1_Cindex_Main/L0Test results_display/Table1_Cindex_Main

# 原始结果让位（mv，重跑前用）；display 始终复制
python results_display/scripts/Backup_Experiment_Results.py --name failed_nodes \
    --move --include results/Table1_Cindex_Main/L0Test/L0_COAD_poe_model_val

# 只预览，不落盘
python results_display/scripts/Backup_Experiment_Results.py --name paper_v1 --dry-run

注意事项：
- 备份目录已存在时自动追加 -2/-3 后缀，不会覆盖既有备份；
- raw 结果体积可能很大（checkpoint/pkl），默认只归档 display，原始结果请显式 --include；
- --move 只对 results/ 下的来源生效；results_display/ 来源始终复制。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

# 官方四大实验的显示产物（--include 缺省时归档这些）
DEFAULT_DISPLAY_TARGETS = [
    "results_display/Table1_Cindex_Main",
    "results_display/Table2_Modality_Missing_Tset",
    "results_display/Table3_Train_MissingRate",
    "results_display/Table4_Abaltion_Test",
]


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def _fmt_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024 or unit == "TB":
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


def _du(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _resolve_jobs(root: Path, include: list[str] | None, move_raw: bool) -> list[tuple[Path, str, str]]:
    """把 --include 路径解析为 (src_abs, 归档相对路径, 模式) 列表。"""
    targets = include if include else DEFAULT_DISPLAY_TARGETS
    jobs: list[tuple[Path, str, str]] = []
    for raw_path in targets:
        src = (root / raw_path).resolve()
        try:
            src.relative_to(root)
        except ValueError:
            raise SystemExit(f"[ERROR] 路径必须在项目根内: {raw_path}")
        if not src.exists():
            raise SystemExit(f"[ERROR] 来源不存在: {raw_path}")
        rel = Path(raw_path)
        if rel.parts[0] == "results_display":
            bucket, mode = "display", "copy"
        elif rel.parts[0] == "results":
            bucket, mode = "raw", ("move" if move_raw else "copy")
        else:
            raise SystemExit(f"[ERROR] 只支持 results/ 或 results_display/ 下的路径: {raw_path}")
        # 去掉第一段（results / results_display），归档内保留相对结构
        inner = str(Path(*rel.parts[1:]))
        jobs.append((src, f"{bucket}/{inner}" if inner else bucket, mode))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description="把当前实验结果快照归档到 results/backup/<日期>_<名称>/")
    parser.add_argument("--name", required=True, help="备份名称后缀，最终目录名 = <YYYYMMDD>_<name>")
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="要归档的路径（相对项目根，results/... 或 results_display/...，可多个）；"
             "缺省归档官方四大实验的 display 目录",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="results/ 下的原始结果用 mv（让位，重跑前用），默认复制；display 始终复制",
    )
    parser.add_argument("--dry-run", action="store_true", help="只列出将执行的复制/移动，不实际落盘")
    args = parser.parse_args()

    root = project_root_from_script()
    jobs = _resolve_jobs(root, args.include, args.move)

    backup_root = root / "results" / "backup" / f"{datetime.now().strftime('%Y%m%d')}_{args.name}"
    final_dir = backup_root
    n = 2
    while final_dir.exists():
        final_dir = Path(f"{backup_root}-{n}")
        n += 1

    total = sum(_du(src) for src, _, _ in jobs)
    print(f"备份目录: {final_dir.relative_to(root)}")
    print(f"模式: raw={'move' if args.move else 'copy'}, display=copy")
    print(f"预估体积: {_fmt_size(total)}（{len(jobs)} 项）")
    for src, dest_rel, mode in jobs:
        print(f"  [{mode}] {src.relative_to(root)} -> {dest_rel}")

    if args.dry_run:
        print("[dry-run] 未落盘")
        return 0

    (final_dir / "raw").mkdir(parents=True, exist_ok=True)
    (final_dir / "display").mkdir(parents=True, exist_ok=True)
    manifest: list[str] = []
    for src, dest_rel, mode in jobs:
        dest = final_dir / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if mode == "move":
            shutil.move(str(src), str(dest))
        elif src.is_dir():
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)
        manifest.append(f"| {mode} | {src.relative_to(root)} | {dest_rel} |")

    list_path = final_dir / "list.md"
    lines = [
        f"# {datetime.now().strftime('%Y%m%d')} 实验结果归档清单",
        "",
        f"- 备份时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 名称：{args.name}",
        f"- 复制方式：raw={'move' if args.move else 'copy'}，display=copy",
        f"- 预估体积：{_fmt_size(total)}",
        "",
        "## 内容",
        "",
        "| 模式 | 来源 | 归档位置 |",
        "|---|---|---|",
        *manifest,
    ]
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✓ 归档完成: {final_dir.relative_to(root)}（list.md 已写入）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
