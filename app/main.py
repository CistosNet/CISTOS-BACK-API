from fastapi import FastAPI, File, HTTPException, UploadFile, Form, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import datetime, timedelta
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from apscheduler.schedulers.background import BackgroundScheduler
from typing import List
import zipfile
import logging
import base64
from io import BytesIO
import re
import cv2
import sys

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Diretórios base ---
# Ajuste para PyInstaller
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).resolve().parent.parent

SEGMENTATION_MODEL_PATH = BASE_DIR / "modelo_yolo/best_segmentador.pt"
CLASSIFICATION_MODEL_PATH = BASE_DIR / "modelo_yolo/best_classificador.pt"

segmentation_model = YOLO(str(SEGMENTATION_MODEL_PATH))
classification_model = YOLO(str(CLASSIFICATION_MODEL_PATH))

# Diretórios de uploads e resultados
UPLOAD_DIR = Path(BASE_DIR) / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

RESULTS_DIR = Path(BASE_DIR) / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# --- Servir static e templates ---
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

@app.get("/index.html", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/analysis.html", response_class=HTMLResponse)
async def analysis(request: Request):
    return templates.TemplateResponse("analysis.html", {"request": request})

@app.get("/history.html", response_class=HTMLResponse)
async def history(request: Request):
    return templates.TemplateResponse("history.html", {"request": request})

# --- Função de limpeza ---
def cleanup_uploads(max_age_min: int = 1):
    now = datetime.now()
    cutoff = now - timedelta(minutes=max_age_min)

    removed = 0
    for file in UPLOAD_DIR.glob("*"):
        if file.is_file():
            mtime = datetime.fromtimestamp(file.stat().st_mtime)
            if mtime < cutoff:
                try:
                    file.unlink()
                    removed += 1
                except Exception as e:
                    logging.error(f"Erro ao remover {file}: {e}")

    if removed > 0:
        logging.info(f"Cleanup: {removed} arquivos removidos da pasta uploads/")

scheduler = BackgroundScheduler()
scheduler.add_job(cleanup_uploads, "interval", minutes=10)
scheduler.start()

@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()

# --- Utils ---
def sanitize_filename(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)

# --- Endpoint de predição ---
@app.post("/predict/policistos")
async def predict_policistos(
    request: Request,
    analysis_name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    safe_analysis_name = sanitize_filename(analysis_name)
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    batch_dir = UPLOAD_DIR / f"{safe_analysis_name}_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    output_dir = batch_dir / "labeled"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for file in files:
        safe_name = Path(file.filename).name
        file_path = batch_dir / safe_name

        with open(file_path, "wb") as f:
            f.write(await file.read())

        if file.filename.endswith(".zip"):
            extract_dir = batch_dir / f"zip_{datetime.now().strftime('%H%M%S')}"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            for img_path in extract_dir.rglob("*.*"):
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue
                processed = await process_single_image(img_path, output_dir)
                if processed:
                    results.append(processed)

        elif file.content_type.startswith("image/"):
            processed = await process_single_image(file_path, output_dir)
            if processed:
                results.append(processed)

        elif file.content_type.startswith("video/"):
            frames_dir = batch_dir / f"frames_{datetime.now().strftime('%H%M%S')}"
            frames_dir.mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Não foi possível abrir o vídeo.")

            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            frame_interval = max(frame_rate, 1)

            frame_count = 0
            saved_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    frame_file = frames_dir / f"frame_{saved_count:05d}.jpg"
                    cv2.imwrite(str(frame_file), frame)
                    processed = await process_single_image(frame_file, output_dir)
                    if processed:
                        results.append(processed)
                    saved_count += 1
                frame_count += 1

            cap.release()

        else:
            raise HTTPException(status_code=400, detail="Formato de arquivo não suportado.")

    if not results:
        raise HTTPException(status_code=400, detail="Nenhuma imagem/vídeo válido processado.")

    zip_filename = f"Resultados_{safe_analysis_name}_{timestamp}.zip"
    zip_path = RESULTS_DIR / zip_filename

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for r in results:
            zipf.write(r["path"], arcname=r["path"].name)

    base_url = str(request.base_url).rstrip("/")
    report_url = f"{base_url}/download/{zip_filename}"

    response = {
        "results": [
            {
                "image": r["image_base64"],
                "info": {"cistos": r["cistos"]}
            }
            for r in results
        ],
        "report_url": report_url
    }

    return JSONResponse(content=response)

# --- Processamento de imagem ---
async def process_single_image(img_path: Path, output_dir: Path):
    cls_results = classification_model.predict(
        source=str(img_path), save=False, imgsz=416, show_labels=False
    )
    probs = cls_results[0].probs
    if probs is None:
        return None

    top1_idx = int(probs.top1)
    class_name = classification_model.names[top1_idx].lower()

    if class_name == "saudavel":
        return None

    seg_results = segmentation_model.predict(
        source=str(img_path), save=False, imgsz=416, show_labels=False
    )
    cisto_count = len(seg_results[0].boxes)

    seg_img = seg_results[0].plot(labels=False)
    img_pil_clean = Image.fromarray(seg_img).convert("RGB")

    img_pil_annotated = img_pil_clean.copy()
    draw = ImageDraw.Draw(img_pil_annotated)
    font = ImageFont.load_default()

    text = f"Cistos: {cisto_count}"
    text_size = draw.textbbox((0, 0), text, font=font)
    text_width = text_size[2] - text_size[0]
    text_height = text_size[3] - text_size[1]

    x, y = 10, 10
    draw.rectangle(
        [x - 2, y - 2, x + text_width + 2, y + text_height + 2],
        fill="black"
    )
    draw.text((x, y), text, font=font, fill="white")

    output_file = output_dir / f"{img_path.stem}_cistos{cisto_count}.jpg"
    img_pil_annotated.save(output_file)

    buffered = BytesIO()
    img_pil_clean.save(buffered, format="PNG")
    img_base64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

    return {
        "path": output_file,
        "image_base64": img_base64,
        "cistos": cisto_count
    }

# --- Download ---
@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    return FileResponse(path=file_path, filename=filename, media_type="application/zip")
