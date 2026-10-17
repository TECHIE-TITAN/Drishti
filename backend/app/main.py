from __future__ import annotations

import io
import json
from typing import Dict, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from . import config, detector, sarvam, signs

app = FastAPI(title="Drishti")

_translation_cache: Dict[Tuple[str, str], str] = {}
_audio_cache: Dict[Tuple[str, str, str], Tuple[str, str]] = {}

EXPLAIN_SYSTEM = (
    "You are Drishti, an accessibility assistant that explains public signs and notice boards. "
    "Use ONLY the text provided; the text came from OCR and may contain errors. "
    "Never invent rules, fines, timings or directions that are not in the text. "
    "If the text is unclear or incomplete, say so honestly. "
    "Reply in {language}, in {style} sentences, at most 4 sentences, no markdown, no lists."
)

ASK_SYSTEM = (
    "You are Drishti, an accessibility assistant. The user is looking at a sign or board. "
    "Answer their question using ONLY the sign information below. If the answer is not there, "
    "say you cannot tell from the sign and suggest asking a person nearby. "
    "Reply in {language}, in {style} sentences, at most 3 sentences, no markdown.\n\n"
    "SIGN INFORMATION:\n{context}"
)


def _style(simple: bool) -> str:
    return "very short, simple, everyday-word" if simple else "clear and short"


def _lang_name(code: str) -> str:
    if code not in config.LANGUAGES:
        raise HTTPException(400, "Unsupported language: %s" % code)
    return config.LANGUAGES[code]


def _translate_cached(text: str, target: str) -> str:
    key = (text, target)
    if key not in _translation_cache:
        _translation_cache[key] = sarvam.translate(text, target)
    return _translation_cache[key]


def _speak(text: str, lang: str, simple: bool) -> dict:
    key = (text, lang, "slow" if simple else "normal")
    if key not in _audio_cache:
        _audio_cache[key] = sarvam.speak(text, lang, pace=0.85 if simple else 1.0)
    audio, mime = _audio_cache[key]
    return {"audio_b64": audio, "audio_mime": mime}


def _prepare_image(raw: bytes) -> bytes:
    """Fix phone rotation and shrink to keep OCR upload fast."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    img.thumbnail((1600, 1600))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=88)
    return out.getvalue()


@app.exception_handler(sarvam.SarvamError)
async def _sarvam_error(_, exc: sarvam.SarvamError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(RuntimeError)
async def _runtime_error(_, exc: RuntimeError):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/api/health")
def health():
    try:
        config.api_key()
        key_ok = True
    except RuntimeError:
        key_ok = False
    return {"ok": True, "api_key_configured": key_ok, "detector": detector.available()}


@app.get("/api/languages")
def languages():
    return config.LANGUAGES


@app.get("/api/signs")
def list_signs():
    return signs.all_signs()


@app.post("/api/analyze")
def analyze(
    image: Optional[UploadFile] = File(None),
    language: str = Form("en-IN"),
    sign_language: str = Form("en-IN"),
    simple: bool = Form(False),
    sign_id: str = Form(""),
):
    """Explain a sign. Either `image` (camera) or `sign_id` (manual pick) is required."""
    lang_name = _lang_name(language)
    raw: Optional[bytes] = None
    entry: Optional[dict] = None
    source = ""
    confidence: Optional[float] = None
    ocr_text = ""

    if sign_id:
        entry = signs.get(sign_id)
        if not entry:
            raise HTTPException(400, "Unknown sign_id")
        source = "picked"
    else:
        if image is None:
            raise HTTPException(400, "Provide an image or a sign_id")
        raw = _prepare_image(image.file.read())

        hit = detector.detect(raw)
        if hit:
            confidence = hit["conf"]
            if hit["entry"]:
                entry, source = hit["entry"], "detector"
                ocr_text = "Detected: %s" % hit["label"]
            else:
                # Detected but not in the verified database: explain from the label only.
                explanation = sarvam.chat(
                    EXPLAIN_SYSTEM.format(language=lang_name, style=_style(simple))
                    + " The text is the name of a traffic sign detected by a camera model, not text printed on it.",
                    "Detected traffic sign: %s" % hit["label"],
                )
                return {
                    "kind": "text_board",
                    "source": "detector (unverified)",
                    "confidence": confidence,
                    "title": hit["label"],
                    "original_text": "Detected: %s" % hit["label"],
                    "explanation": explanation,
                    "context": "Detected traffic sign: %s" % hit["label"],
                    **_speak(explanation, language, simple),
                }
        if entry is None:
            ocr_text = sarvam.ocr_image(raw, "sign.jpg", "image/jpeg", sign_language)
            if ocr_text:
                entry = signs.match_text(ocr_text)
                source = "ocr+database" if entry else "ocr"

    # 1) Verified database sign: translate the fixed English text, no LLM involved.
    if entry:
        title = _translate_cached(entry["name"], language)
        explanation = _translate_cached("%s %s" % (entry["meaning"], entry["action"]), language)
        context_en = signs.english_text(entry)
        return {
            "kind": "db_sign",
            "source": source,
            "confidence": confidence,
            "title": title,
            "category": entry["category"],
            "original_text": ocr_text,
            "explanation": explanation,
            "context": context_en,
            **_speak("%s. %s" % (title, explanation), language, simple),
        }

    # 2) Free-form text board: ground the LLM on the OCR text.
    if ocr_text:
        explanation = sarvam.chat(
            EXPLAIN_SYSTEM.format(language=lang_name, style=_style(simple)),
            "Sign text:\n%s" % ocr_text[:3000],
        )
        return {
            "kind": "text_board",
            "source": source,
            "title": "",
            "original_text": ocr_text,
            "explanation": explanation,
            "context": ocr_text[:3000],
            **_speak(explanation, language, simple),
        }

    # 3) Nothing readable.
    msg = _translate_cached(
        "I could not read any sign. Please move closer, hold the phone steady, and try again.", language
    )
    return {
        "kind": "none",
        "source": source,
        "title": "",
        "original_text": "",
        "explanation": msg,
        "context": "",
        **_speak(msg, language, simple),
    }


@app.post("/api/ask")
def ask(
    audio: Optional[UploadFile] = File(None),
    text: str = Form(""),
    context: str = Form(""),
    language: str = Form("en-IN"),
    simple: bool = Form(False),
):
    """Answer a spoken (audio) or typed (text) question about the current sign."""
    lang_name = _lang_name(language)
    question = text.strip()
    if audio is not None and not question:
        data = audio.file.read()
        if not data:
            raise HTTPException(400, "Empty audio")
        question = sarvam.transcribe(data, audio.filename or "question.webm", audio.content_type or "audio/webm")
    if not question:
        raise HTTPException(400, "I could not hear a question. Please try again.")

    answer = sarvam.chat(
        ASK_SYSTEM.format(language=lang_name, style=_style(simple), context=context or "(no sign scanned yet)"),
        question,
    )
    return {"question": question, "answer": answer, **_speak(answer, language, simple)}


# Frontend (must be mounted last so /api routes win).
app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")
