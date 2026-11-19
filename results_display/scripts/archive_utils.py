"""展示层旧产物归档工具。

项目规则：不直接删除旧展示产物；统一移动到
`AnysoleWorkspace/_trash/<YYYY-MM-DD>/<原相对路径>`，同一天重复归档时
加时间戳避免覆盖。Table4 的两个展示脚本已有等价实现，这里抽出共享版本，
供 L1 展示流程中的 Table1 / Table3 / FigC 复用。
"""

from __future__ import annotations

import datetime
import shutil
from pathlib import Path


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def trash_root() -> Path:
    return (
        project_root_from_script()
        / "AnysoleWorkspace"
        / "_trash"
        / datetime.date.today().isoformat()
    )


def archive_path(path: Path) -> Path | None:
    """把旧产物移到 AnysoleWorkspace/_trash/<date>/，不删除。

    返回归档目标；源文件不存在时返回 None。
    """
    path = Path(path)
    if not path.exists():
        return None

    project_root = project_root_from_script()
    try:
        relative = path.relative_to(project_root)
    except ValueError:
        relative = Path(path.name)

    target = trash_root() / relative
    if target.exists():
        stamp = datetime.datetime.now().strftime("%H%M%S%f")
        if target.is_dir():
            target = target.with_name(f"{target.name}.{stamp}")
        else:
            target = target.with_name(f"{target.stem}.{stamp}{target.suffix}")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    print(f"[ARCHIVE] {path} -> {target}")
    return target


def list_existing_artifacts(output_dir: Path) -> list[Path]:
    """列出 output_dir 下所有将被覆盖的展示产物（递归：csv/png/pdf）。"""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return []
    paths = (
        set(output_dir.rglob("*.csv"))
        | set(output_dir.rglob("*.png"))
        | set(output_dir.rglob("*.pdf"))
    )
    return sorted(paths)


def archive_existing_artifacts(
    output_dir: Path,
    *,
    confirm: bool,
    label: str = "Table3",
) -> list[Path]:
    """先列完整清单；confirm=False 时只列清单并退出，不覆盖任何旧文件。"""
    existing = list_existing_artifacts(output_dir)
    if not existing:
        return []

    print(f"[INVENTORY] {label}: {len(existing)} existing artifact(s) under {output_dir}")
    for path in existing:
        print(f"  - {path}")

    if not confirm:
        raise SystemExit(
            f"[ERROR] {label}: refusing to overwrite existing artifacts without confirmation. "
            "Re-run with --confirm-archive to move them to AnysoleWorkspace/_trash/<date>/ first."
        )

    for path in existing:
        archive_path(path)
    return existing
