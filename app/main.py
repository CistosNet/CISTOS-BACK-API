from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from datetime import datetime, timedelta
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from apscheduler.schedulers.background import BackgroundScheduler
import io
import shutil
import logging

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent

SEGMENTATION_MODEL_PATH = BASE_DIR / "modelos_yolo/best_segmentador.pt"
CLASSIFICATION_MODEL_PATH = BASE_DIR / "modelos_yolo/best_classificador.pt"

segmentation_model = YOLO(str(SEGMENTATION_MODEL_PATH))
classification_model = YOLO(str(CLASSIFICATION_MODEL_PATH))

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# -----------------------------
# LIMPEZA AUTOMÁTICA DE UPLOADS
# -----------------------------
def cleanup_uploads(max_age_hours: int = 1):
    """Remove arquivos mais antigos que max_age_hours da pasta uploads/"""
    now = datetime.now()
    cutoff = now - timedelta(hours=max_age_hours)

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

# Scheduler em background
scheduler = BackgroundScheduler()
scheduler.add_job(cleanup_uploads, "interval", hours=1)  # roda a cada 1h
scheduler.start()

@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()


# -----------------------------
# ROTAS BÁSICAS
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "rota 01"}


@app.get("/coffee")
def coffee():
    return JSONResponse(status_code=418, content={"message": "I'm a Teapot"})


# -----------------------------
# ROTAS DE SEGMENTAÇÃO
# -----------------------------
@app.post("/images/predict/segmentation")
async def upload_image(file: UploadFile = File(...)):
    if file is None:
        return JSONResponse(content={"error": "Nenhum arquivo recebido"}, status_code=400)

    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    file_path = UPLOAD_DIR / filename

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    results = segmentation_model.predict(source=str(file_path), save=False, imgsz=640)

    segmented_img = results[0].plot()

    img_pil = Image.fromarray(segmented_img)
    img_byte_arr = io.BytesIO()
    img_pil.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    return StreamingResponse(content=img_byte_arr, media_type="image/jpeg")


@app.post("/videos/predict/segmentation")
async def upload_video(file: UploadFile = File(...)):

    if not file.filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
        return JSONResponse(
            content={"error": "Formato de vídeo não suportado"}, status_code=400
        )

    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)  # type: ignore

    results = segmentation_model.predict(
        source=str(file_path),
        save=True,
        project=UPLOAD_DIR,
        name="processed_seg",
        vid_stride=1,
    )

    save_dir = results[0].save_dir

    output_videos = list(Path(save_dir).rglob("*.*"))
    output_videos = [
        f for f in output_videos if f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]
    ]

    if not output_videos:
        return JSONResponse(
            content={"error": "Vídeo processado não encontrado"}, status_code=500
        )

    output_video = output_videos[0]

    return FileResponse(
        path=output_video,
        media_type=f"video/{output_video.suffix.lstrip('.')}",
        filename=output_video.name,
    )


# -----------------------------
# ROTAS DE CLASSIFICAÇÃO
# -----------------------------
@app.post("/images/predict/classification")
async def classify_image(file: UploadFile = File(...)):
    if file is None:
        return JSONResponse(content={"error": "Nenhum arquivo recebido"}, status_code=400)

    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    file_path = UPLOAD_DIR / filename

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    results = classification_model.predict(source=str(file_path), save=False, imgsz=640)

    probs = results[0].probs  # probabilidades
    if probs is None:
        return JSONResponse(content={"error": "Falha ao classificar a imagem"}, status_code=500)

    top1_idx = int(probs.top1)
    top1_conf = float(probs.top1conf)
    class_name = classification_model.names[top1_idx]

    img_pil = Image.open(file_path).convert("RGB")
    width, height = img_pil.size

    text = f"{class_name} ({top1_conf:.2f})"
    font = ImageFont.load_default()

    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    text_bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    new_height = height + text_height + 20
    new_img = Image.new("RGB", (width, new_height), color="black")
    new_img.paste(img_pil, (0, 0))

    draw = ImageDraw.Draw(new_img)
    text_x = (width - text_width) // 2
    text_y = height + 10
    draw.text((text_x, text_y), text, fill="white", font=font)

    img_byte_arr = io.BytesIO()
    new_img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    return StreamingResponse(content=img_byte_arr, media_type="image/jpeg")


@app.post("/videos/predict/classification")
async def classify_video(file: UploadFile = File(...)):

    if not file.filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
        return JSONResponse(
            content={"error": "Formato de vídeo não suportado"}, status_code=400
        )

    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)  # type: ignore

    results = classification_model.predict(
        source=str(file_path),
        save=True,
        project=UPLOAD_DIR,
        name="processed_cls",
        vid_stride=5,  # processa a cada 5 frames
    )

    save_dir = results[0].save_dir

    output_videos = list(Path(save_dir).rglob("*.*"))
    output_videos = [
        f for f in output_videos if f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]
    ]

    if not output_videos:
        return JSONResponse(
            content={"error": "Vídeo classificado não encontrado"}, status_code=500
        )

    output_video = output_videos[0]

    return FileResponse(
        path=output_video,
        media_type=f"video/{output_video.suffix.lstrip('.')}",
        filename=output_video.name,
    )
