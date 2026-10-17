"""Download the pretrained sign detector and print its class names.

Run from the project root:
    backend/.venv/bin/pip install ultralytics huggingface_hub
    backend/.venv/bin/python backend/scripts/fetch_model.py

Then edit backend/data/class_map.json for any class the fuzzy matcher misses.
"""
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
REPO, FILE = "ankitjha07/Traffic-Sign-Detection-YOLOv8", "best.pt"

path = hf_hub_download(REPO, FILE, local_dir=str(ROOT / "backend" / "models"))
print("weights:", path)

names = YOLO(path).names
print("\nModel classes:")
for i, n in names.items():
    print("  %2d  %s" % (i, n))

import sys

sys.path.insert(0, str(ROOT / "backend"))
from app import detector  # noqa: E402

print("\nAuto-mapping to the sign database:")
unmapped = []
for n in names.values():
    entry = detector.map_label(str(n))
    print("  %-30s -> %s" % (n, entry["id"] if entry else "(unmapped, will be explained from label only)"))
    if not entry:
        unmapped.append(str(n))

cm = ROOT / "backend" / "data" / "class_map.json"
if not cm.exists():
    cm.write_text(json.dumps({}, indent=2))
print("\nAdd overrides to %s like {\"<class name>\": \"<sign id>\"}." % cm)
