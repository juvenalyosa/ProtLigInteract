#!/usr/bin/env python3
"""
Build a PyMOL-installable plugin zip with a top-level folder.

Usage:
  python tools/build_zip.py                # builds dist/ProtLigInteract.zip
  python tools/build_zip.py --name MyPlug  # custom folder name inside zip
  python tools/build_zip.py --out my.zip   # custom output path
  python tools/build_zip.py --no-shots     # skip screenshots
"""

import argparse
import os
import shutil
import sys
import zipfile
from glob import glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

FILES_MANDATORY = ["__init__.py", "code_v2.py", "protliginteract.ui"]
FILES_OPTIONAL = ["README.md", "requirements.txt", "CONTRIBUTING.md"]


def build_zip(top_name: str, out_zip: str, include_shots: bool = True) -> None:
    stage_root = os.path.join(ROOT, "build", top_name)
    if os.path.exists(stage_root):
        shutil.rmtree(stage_root)
    os.makedirs(stage_root, exist_ok=True)

    # Copy core files
    for fname in FILES_MANDATORY + FILES_OPTIONAL:
        src = os.path.join(ROOT, fname)
        if os.path.isfile(src):
            shutil.copy2(src, stage_root)

    # Optional screenshots
    if include_shots:
        shots_dir = os.path.join(ROOT, "screenshots")
        if os.path.isdir(shots_dir):
            dst = os.path.join(stage_root, "screenshots")
            os.makedirs(dst, exist_ok=True)
            for pat in ("*.png", "*.jpg", "*.jpeg"):
                for img in glob(os.path.join(shots_dir, pat)):
                    shutil.copy2(img, dst)

    # Ensure __init__.py exists in stage folder
    if not os.path.isfile(os.path.join(stage_root, "__init__.py")):
        print("ERROR: __init__.py not found. Run from repo root and ensure files exist.", file=sys.stderr)
        sys.exit(1)

    # Create dist dir
    dist_dir = os.path.join(ROOT, "dist")
    os.makedirs(dist_dir, exist_ok=True)
    out_path = os.path.abspath(out_zip if os.path.isabs(out_zip) else os.path.join(ROOT, out_zip))
    if not out_path.endswith(".zip"):
        out_path += ".zip"

    # Build zip
    if os.path.isfile(out_path):
        os.remove(out_path)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(stage_root):
            for f in files:
                abspath = os.path.join(root, f)
                rel = os.path.relpath(abspath, os.path.join(stage_root, os.pardir))
                # rel should start with top_name/
                zf.write(abspath, arcname=rel)
    print(f"Created {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Build ProtLigInteract PyMOL plugin zip")
    ap.add_argument("--name", default="ProtLigInteract", help="Top-level folder name inside the zip")
    ap.add_argument(
        "--out",
        default=os.path.join("dist", "ProtLigInteract.zip"),
        help="Output zip path (default: dist/ProtLigInteract.zip)",
    )
    ap.add_argument("--no-shots", action="store_true", help="Do not include screenshots")
    args = ap.parse_args()

    build_zip(args.name, args.out, include_shots=not args.no_shots)


if __name__ == "__main__":
    main()

