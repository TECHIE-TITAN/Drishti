"""Thin wrappers around the Sarvam APIs used by Drishti."""
from __future__ import annotations

import base64
import io
import re
import time
import zipfile
from typing import List, Optional, Tuple

import httpx
import requests
from sarvamai import SarvamAI

from . import config


class SarvamError(Exception):
    pass


def _headers() -> dict:
    return {"api-subscription-key": config.api_key()}


def _check(r: httpx.Response, what: str) -> dict:
    if r.status_code >= 400:
        raise SarvamError("%s failed (%s): %s" % (what, r.status_code, r.text[:300]))
    return r.json()


def _sdk() -> SarvamAI:
    return SarvamAI(api_subscription_key=config.api_key())


# ---------------------------------------------------------------- OCR (Sarvam Vision)
def _html_to_text(s: str) -> str:
    s = re.sub(r"<(script|style).*?</\1>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<br\s*/?>|</p>|</div>|</h\d>|</li>|</tr>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", s)  # markdown images
    s = s.replace("&nbsp;", " ").replace("&amp;", "&")
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"\n\s*\n+", "\n", s).strip()


def ocr_image(image: bytes, filename: str, content_type: str, language: str = "en-IN",
              timeout: float = 90.0) -> str:
    """Run Sarvam Vision (Document Intelligence) on one image and return plain text."""
    client = _sdk()
    try:
        job = client.doc_ai.digitise(file=[(filename, io.BytesIO(image), content_type)], language=language)
        deadline = time.time() + timeout
        status = job.status
        while str(status).lower() not in ("completed", "partially_completed", "failed", "rejected"):
            if time.time() > deadline:
                raise SarvamError("OCR timed out")
            time.sleep(1.0)
            status = client.doc_ai.get_status(job_id=job.job_id).status
        if str(status).lower() not in ("completed", "partially_completed"):
            raise SarvamError("OCR job ended with status %s" % status)
        url = client.doc_ai.get_download_url(job_id=job.job_id).url
    except SarvamError:
        raise
    except Exception as e:  # SDK ApiError, network errors
        raise SarvamError("OCR failed: %s" % e)

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    texts: List[str] = []
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        for name in sorted(z.namelist()):
            low = name.lower()
            if low.startswith("metadata/") or low.endswith("manifest.json"):
                continue
            if low.endswith((".md", ".html", ".htm", ".txt")):
                texts.append(_html_to_text(z.read(name).decode("utf-8", "ignore")))
    return "\n".join(t for t in texts if t).strip()


# ---------------------------------------------------------------- Translate (Mayura)
def translate(text: str, target: str, source: str = "en-IN") -> str:
    if not text or target == source:
        return text
    r = httpx.post(
        config.SARVAM_BASE + "/translate",
        headers=_headers(),
        json={
            "input": text[:1000],
            "source_language_code": source,
            "target_language_code": target,
            "speaker_gender": "Female",
        },
        timeout=30,
    )
    return _check(r, "translate")["translated_text"]


# ---------------------------------------------------------------- Chat (Sarvam-105B)
def chat(system: str, user: str, max_tokens: int = 800) -> str:
    r = httpx.post(
        config.SARVAM_BASE + "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": config.CHAT_MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "reasoning_effort": None,  # no thinking tokens: lower latency, nothing to strip
        },
        timeout=60,
    )
    data = _check(r, "chat")
    content = (data["choices"][0]["message"].get("content") or "").strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    if not content:
        raise SarvamError("chat returned an empty answer")
    return content


# ---------------------------------------------------------------- STT (Saaras)
def transcribe(audio: bytes, filename: str, content_type: str) -> str:
    r = httpx.post(
        config.SARVAM_BASE + "/speech-to-text",
        headers=_headers(),
        files={"file": (filename, audio, content_type)},
        data={"model": config.STT_MODEL, "mode": "transcribe"},
        timeout=60,
    )
    return (_check(r, "speech-to-text").get("transcript") or "").strip()


# ---------------------------------------------------------------- TTS (Bulbul)
def _clip(text: str, limit: int = 2400) -> str:
    """REST TTS accepts 2500 chars; cut at a sentence boundary."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    m = max(cut.rfind(". "), cut.rfind("।"), cut.rfind("? "), cut.rfind("! "))
    return cut[: m + 1] if m > 200 else cut


def speak(text: str, language: str, pace: float = 1.0) -> Tuple[str, str]:
    """Returns (base64 audio, mime type)."""
    r = httpx.post(
        config.SARVAM_BASE + "/text-to-speech",
        headers=_headers(),
        json={
            "text": _clip(text),
            "language_code": language,
            "model": config.TTS_MODEL,
            "speaker": config.TTS_SPEAKER,
            "pace": pace,
        },
        timeout=60,
    )
    audios = _check(r, "text-to-speech").get("audios")
    if isinstance(audios, list):
        audios = audios[0] if audios else None
    if not audios:
        raise SarvamError("text-to-speech returned no audio")
    base64.b64decode(audios[:64] + "=" * (-len(audios[:64]) % 4))  # sanity check it is base64
    return audios, "audio/wav"
