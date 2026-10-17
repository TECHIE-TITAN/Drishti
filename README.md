# Drishti — multilingual sign & board assistant

Drishti is an accessibility assistant built on Sarvam AI. A user points a phone at a traffic sign or public notice board and gets the meaning in their own language, as large captions and spoken audio. They can then ask follow-up questions by voice or text. It is aimed at people who are blind or low-vision, deaf or hard of hearing, low-literacy, or unfamiliar with the local language.

## Workflow

1. **Choose languages.** The user picks the language to be explained in, and the language written on the sign.
2. **Capture.** They take a photo of the sign (live camera, or the phone's native camera picker). Pictorial signs with no text can also be picked from a list.
3. **Identify the sign.**
   - A pretrained YOLO detector recognises pictorial traffic signs (optional, see the note at the bottom).
   - Otherwise Sarvam Vision OCR reads the text on the sign.
4. **Explain.**
   - If the sign matches the verified database (`backend/data/signs.json`), its fixed English explanation is translated with Sarvam Mayura. The LLM never rewrites it.
   - If it is a free-form text board, Sarvam-105B explains it, restricted to the text that was read.
5. **Speak.** Sarvam Bulbul reads the explanation aloud. The same text is shown as large captions. A "simple words, slower voice" option is available.
6. **Ask.** The user asks about the sign by voice (Sarvam Saaras speech-to-text) or by typing. Sarvam-105B answers using only that sign's information, and Bulbul speaks the answer.

```
photo ─► detector (optional YOLO) ─┐
      └► Sarvam Vision OCR ─► sign DB match ─► Mayura translate (verified text)
                                     └ no match ─► Sarvam-105B (grounded on OCR text)
                       ─► Bulbul TTS ─► audio + large captions
voice question ─► Saaras STT ─► Sarvam-105B (grounded on current sign) ─► Bulbul
```

## Stack

- **Backend:** FastAPI (Python), serving the API and the web UI.
- **Frontend:** a single mobile-first, accessible HTML page (`frontend/index.html`).
- **Sarvam APIs:** Vision (OCR), Mayura (translation), Sarvam-105B (chat), Saaras (speech-to-text), Bulbul (text-to-speech).
- **Detector (optional):** YOLO via `ultralytics`.

## Run

```bash
cd "SARVAM X IIIT Hyd"
# one-time setup (already done if backend/.venv exists):
#   python3 -m venv backend/.venv && backend/.venv/bin/pip install -r backend/requirements.txt
# put your key in .env  ->  SARVAM_API_KEY=...
backend/.venv/bin/uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
```

Open http://localhost:8000. On a phone (same Wi-Fi) open `http://<laptop-ip>:8000`. Live camera and microphone capture need HTTPS, so over plain HTTP the app falls back to the phone's native camera and recorder pickers. For live capture use a tunnel such as `ngrok http 8000`.

## Files

- `backend/app/main.py` — API (`/api/analyze`, `/api/ask`, `/api/signs`, `/api/languages`, `/api/health`)
- `backend/app/sarvam.py` — Sarvam calls (Vision OCR, Mayura, Sarvam-105B, Saaras, Bulbul)
- `backend/app/signs.py`, `backend/app/detector.py` — sign database matching and the optional YOLO detector
- `backend/data/signs.json` — verified sign database; `backend/data/class_map.json` — detector class-to-sign overrides
- `backend/scripts/fetch_model.py` — downloads the pretrained detector weights and prints its classes
- `frontend/index.html` — the UI

## Optional: enable the pictorial sign detector

Uses [ankitjha07/Traffic-Sign-Detection-YOLOv8](https://huggingface.co/ankitjha07/Traffic-Sign-Detection-YOLOv8) (MIT, 36 classes, training dataset not stated).

```bash
backend/.venv/bin/pip install -r backend/requirements.txt
backend/.venv/bin/python backend/scripts/fetch_model.py   # saves backend/models/best.pt, prints classes + auto-mapping
```

Fix any unmapped classes in `backend/data/class_map.json` (`{"<class name>": "<sign id>"}`). Unmapped detections are still explained from their label and marked "unverified".

## Note: current state of image detection

- **Sarvam Vision is used for OCR only.** It reads text printed on signs and boards. It is a document model and does **not** recognise symbols, so a symbol-only sign (for example a No Entry or School Ahead pictogram with no words) is not identified by it.
- **The YOLO detector is integrated but not enabled by default.** It activates only when `backend/models/best.pt` exists, and the weights have not been downloaded or tested in this repository. Until then the app reports `detector: false`.
- **Without the detector,** text signs work through OCR, and pictorial signs must be chosen with the "Pick it from the list" dropdown.
- **Even with the detector,** the pretrained model is not trained specifically on Indian signs, so accuracy on Indian roads is unverified, and its class names may need mapping in `class_map.json`.
- **Not yet tested end to end:** OCR, speech-to-text and the LLM routes were not exercised against the live API at the time of writing. Only the sign-picker path (translation and text-to-speech) was confirmed working.
