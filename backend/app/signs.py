"""Verified sign database and text matching."""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from . import config

_SIGNS: List[dict] = json.loads((config.DATA_DIR / "signs.json").read_text(encoding="utf-8"))
_BY_ID: Dict[str, dict] = {s["id"]: s for s in _SIGNS}


def raw() -> List[dict]:
    return _SIGNS


def all_signs() -> List[dict]:
    return [{"id": s["id"], "name": s["name"], "category": s["category"]} for s in _SIGNS]


def get(sign_id: str) -> Optional[dict]:
    return _BY_ID.get(sign_id)


def _norm(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return text.strip()


def match_text(text: str) -> Optional[dict]:
    """Match OCR text against sign keywords.

    A keyword starting with '=' must equal the whole (normalised) text, which stops
    'STOP' matching 'BUS STOP'. Other keywords match as whole words/phrases; the longest wins.
    """
    norm = _norm(text)
    if not norm:
        return None
    best, best_len = None, 0
    padded = " %s " % norm
    for s in _SIGNS:
        for kw in s["keywords"]:
            if kw.startswith("="):
                k = _norm(kw[1:])
                hit = norm == k
            else:
                k = _norm(kw)
                hit = (" %s " % k) in padded
            if hit and len(k) > best_len:
                best, best_len = s, len(k)
    return best


def english_text(sign: dict) -> str:
    return "%s. %s %s" % (sign["name"], sign["meaning"], sign["action"])
