"""Pretrained pictorial traffic-sign detector (YOLO / ultralytics).

Default weights: ankitjha07/Traffic-Sign-Detection-YOLOv8 (MIT), downloaded to
backend/models/best.pt by `python backend/scripts/fetch_model.py`.

Sarvam Vision is a document OCR model, not a traffic-sign detector, so pictorial signs
go through this model. Its class names are mapped to entries of data/signs.json by:
  1. data/class_map.json  ({"<model class name>": "<sign id>"}), then
  2. fuzzy matching of the class name against sign ids, names and keywords.
Classes that map to nothing are still reported (label only) and explained by the LLM
from the label text, flagged as unverified.
"""
from __future__ import annotations

import io
import json
import re
from typing import Optional

from . import config, signs

DEFAULT_WEIGHTS = config.ROOT / "backend" / "models" / "best.pt"
CLASS_MAP_FILE = config.DATA_DIR / "class_map.json"

_model = None
_load_failed = False


def _weights_path() -> Optional[str]:
    if config.SIGN_MODEL_PATH:
        return config.SIGN_MODEL_PATH
    return str(DEFAULT_WEIGHTS) if DEFAULT_WEIGHTS.exists() else None


def available() -> bool:
    return _weights_path() is not None and not _load_failed


def _load():
    global _model, _load_failed
    if _model is None and not _load_failed:
        try:
            from ultralytics import YOLO  # optional dependency

            _model = YOLO(_weights_path())
        except Exception as e:
            print("[detector] disabled:", e)
            _load_failed = True
    return _model


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _class_map() -> dict:
    try:
        return json.loads(CLASS_MAP_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def map_label(label: str) -> Optional[dict]:
    """Map a model class name to a verified sign entry, or None."""
    explicit = _class_map().get(label)
    if explicit:
        return signs.get(explicit)
    n = _norm(label)
    if not n:
        return None
    best, best_len = None, 0
    for s in signs.raw():
        candidates = [s["id"].replace("_", " "), s["name"]] + [k for k in s["keywords"] if not k.startswith("=")]
        for c in candidates:
            c = _norm(c)
            # exact match, or the model label contains the full sign phrase ("speed limit 50")
            if c and (n == c or (" %s " % c) in (" %s " % n)) and len(c) > best_len:
                best, best_len = s, len(c)
    return best


def class_names() -> dict:
    model = _load()
    return dict(model.names) if model is not None else {}


def detect(image: bytes, min_conf: float = 0.4) -> Optional[dict]:
    """Most confident detection as {"label", "conf", "entry"} (entry may be None), else None."""
    if not available():
        return None
    model = _load()
    if model is None:
        return None
    from PIL import Image

    img = Image.open(io.BytesIO(image)).convert("RGB")
    result = model.predict(img, verbose=False)[0]
    best = None
    for box in result.boxes:
        conf = float(box.conf[0])
        if conf >= min_conf and (best is None or conf > best["conf"]):
            label = str(result.names[int(box.cls[0])])
            best = {"label": label, "conf": conf}
    if best:
        best["entry"] = map_label(best["label"])
    return best
