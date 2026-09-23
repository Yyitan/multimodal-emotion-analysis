#!/usr/bin/env python3
"""Read-only audit of explanation-set PKL modality availability and video duration."""
import argparse
import glob
import json
import pickle
import subprocess
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.explain import inspect_modalities


def load_pickle(path):
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except ModuleNotFoundError as error:
        # Newer NumPy pickles may name numpy._core while old installations expose numpy.core.
        if not error.name.startswith("numpy._core"):
            raise
        import sys
        import numpy.core as core
        sys.modules.setdefault("numpy._core", core)
        for sub in ("multiarray", "numeric", "umath", "_multiarray_umath"):
            try:
                sys.modules.setdefault("numpy._core." + sub, __import__("numpy.core." + sub, fromlist=["*"]))
            except ImportError:
                pass
        with open(path, "rb") as handle:
            return pickle.load(handle)


def duration(path):
    try:
        value = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)], text=True
        ).strip()
        return float(value)
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None


def audit(version_dir):
    root = Path(version_dir); records = []
    for name in sorted(glob.glob(str(root / "*.pkl"))):
        sample = load_pickle(name); quality = inspect_modalities(sample)
        sample_id = str(sample.get("id", Path(name).stem))
        records.append({
            "sample_id": sample_id,
            "feature_file": str(Path(name).resolve()),
            "video_file": str((root / "videos" / (sample_id + ".mp4")).resolve()),
            "video_duration_sec": duration(root / "videos" / (sample_id + ".mp4")),
            "modality_availability": quality["availability"],
            "quality_flags": quality["quality_flags"],
            "details": quality["details"],
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("version_dir", help="只读的可解释性评测集对齐版本或未对齐版本目录")
    parser.add_argument("--output", help="可选JSON输出路径；省略则打印到标准输出")
    args = parser.parse_args(); result = audit(args.version_dir)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
