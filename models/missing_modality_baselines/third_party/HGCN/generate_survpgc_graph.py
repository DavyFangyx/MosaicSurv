#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path


TARGET = Path(__file__).resolve().parents[4] / "SurvPGC_Workspace" / "generate_hgcn_graph.py"


if __name__ == "__main__":
    print(f"[HGCN] generate_survpgc_graph.py moved to {TARGET}")
    runpy.run_path(str(TARGET), run_name="__main__")

