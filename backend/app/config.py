import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

SARVAM_BASE = "https://api.sarvam.ai"
CHAT_MODEL = os.getenv("CHAT_MODEL", "sarvam-105b")
STT_MODEL = os.getenv("STT_MODEL", "saaras:v4")
TTS_MODEL = os.getenv("TTS_MODEL", "bulbul:v3")
TTS_SPEAKER = os.getenv("TTS_SPEAKER", "shubh")
SIGN_MODEL_PATH = os.getenv("SIGN_MODEL_PATH", "").strip()
FRONTEND_DIR = ROOT / "frontend"
DATA_DIR = ROOT / "backend" / "data"

# Languages Bulbul v3 can speak (and Mayura can translate to).
LANGUAGES = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "bn-IN": "Bengali",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "gu-IN": "Gujarati",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "pa-IN": "Punjabi",
    "od-IN": "Odia",
}


def api_key() -> str:
    key = os.getenv("SARVAM_API_KEY", "").strip()
    if not key or key == "API_KEY":
        raise RuntimeError("SARVAM_API_KEY is not set. Put your key in .env and restart the server.")
    return key
