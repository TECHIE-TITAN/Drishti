# Drishti — multilingual sign & board assistant

Point a phone at a sign or notice board. Drishti reads it, explains it in your language, speaks it aloud, and answers spoken or typed follow-up questions.

```
photo ─► detector (optional YOLO) ─┐
      └► Sarvam Vision OCR ─► sign DB match ─► Mayura translate (verified text)
                                     └ no match ─► Sarvam-105B (grounded on OCR text)
                       ─► Bulbul TTS ─► audio + large captions
voice question ─► Saaras STT ─► Sarvam-105B (grounded on current sign) ─► Bulbul
```

- Verified database signs (`backend/data/signs.json`) are never rewritten by the LLM; only translated.
- Free-form text boards are explained by the LLM, told to use only the OCR text.
- Pictorial signs need either your own YOLO weights (`SIGN_MODEL_PATH`, class names = sign ids) or the manual picker in the UI.

## Run
```bash
cd "SARVAM X IIIT Hyd"
# put your key in .env  ->  SARVAM_API_KEY=...
backend/.venv/bin/uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000 --reload
```
Open http://localhost:8000. On a phone (same Wi-Fi) open `http://<laptop-ip>:8000`; the camera/mic live capture needs HTTPS, so over plain HTTP the app falls back to the phone's native camera/recorder pickers. For live capture use a tunnel such as `ngrok http 8000`.

## Files
- `backend/app/main.py` API (`/api/analyze`, `/api/ask`, `/api/signs`, `/api/languages`, `/api/health`)
- `backend/app/sarvam.py` Sarvam calls (Vision OCR, Mayura, Sarvam-105B, Saaras, Bulbul)
- `backend/app/signs.py`, `detector.py` sign DB matching, optional YOLO
- `frontend/index.html` mobile-first accessible UI

## Pictorial sign detector (pretrained)
Uses [ankitjha07/Traffic-Sign-Detection-YOLOv8](https://huggingface.co/ankitjha07/Traffic-Sign-Detection-YOLOv8) (MIT, 36 classes, trained on a non-India-specific dataset, so check accuracy on Indian signs).
```bash
backend/.venv/bin/pip install ultralytics huggingface_hub
backend/.venv/bin/python backend/scripts/fetch_model.py   # downloads backend/models/best.pt, prints classes + auto-mapping
```
Fix any unmapped classes in `backend/data/class_map.json` (`{"<class name>": "<sign id>"}`). Unmapped detections are still explained from their label, marked "unverified".
