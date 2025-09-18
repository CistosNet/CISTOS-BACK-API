from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from datetime import datetime, timedelta
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from apscheduler.schedulers.background import BackgroundScheduler
import io
import shutil
import logging
from typing import List
import zipfile
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
BASE_DIR = Path(__file__).parent

@app.get("/frontend")
def frontend():
    return FileResponse(BASE_DIR / "teste.html")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # libera só pro teu front
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE_DIR = Path(__file__).resolve().parent.parent

SEGMENTATION_MODEL_PATH = BASE_DIR / "modelos_yolo_2/best_segmentador.pt"
CLASSIFICATION_MODEL_PATH = BASE_DIR / "modelos_yolo_2/best_classificador.pt"

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

    results = segmentation_model.predict(source=str(file_path), save=False, imgsz=640, show_labels=False)

    # contagem de cistos detectados
    cyst_count = len(results[0].boxes)

    # imagem segmentada
    segmented_img = results[0].plot()
    img_pil = Image.fromarray(segmented_img).convert("RGBA")

    # desenhar texto na imagem
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("arial.ttf", 18)  # menor e mais discreto
    except:
        font = ImageFont.load_default()

    text = f"Cistos: {cyst_count}"

    # bounding box do texto
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # posição mais colada no canto
    x = img_pil.width - text_width - 8
    y = 8

    # fundo preto bem suave
    box_coords = [(x - 4, y - 2), (x + text_width + 4, y + text_height + 2)]
    draw.rectangle(box_coords, fill=(0, 0, 0, 100))  # alpha menor (100)

    # texto cinza claro (mais suave que branco)
    draw.text((x, y), text, font=font, fill=(220, 220, 220, 255))

    # salvar em buffer
    img_byte_arr = io.BytesIO()
    img_pil = img_pil.convert("RGB")
    img_pil.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    return StreamingResponse(
        content=img_byte_arr,
        media_type="image/jpeg",
        headers={"X-Cyst-Count": str(cyst_count)}
    )

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
        show_labels=False,
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
# ROTAS DE CLASSIFICAÇÃO ÚNICA NOVAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# ----------------------------- BATCH

@app.post("/predict/segmentation/Count")
async def predict_segmentation(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    content_type = file.content_type
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    file_path = UPLOAD_DIR / filename

    # Salvar arquivo
    with open(file_path, "wb") as f:
        f.write(await file.read())

    if content_type.startswith("image/"):
        # Processar imagem
        results = segmentation_model.predict(source=str(file_path), save=False, imgsz=640, show_labels=False)

        # Contar cistos detectados
        cisto_count = len(results[0].boxes)

        # Renderizar imagem segmentada
        segmented_img = results[0].plot(labels=False)
        img_pil = Image.fromarray(segmented_img)

        # Adicionar contador no canto superior direito
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.load_default()
        text = f"Cistos: {cisto_count}"

        # Usar textbbox no lugar de textsize
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        draw.rectangle(
            [(img_pil.width - text_w - 10, 10), (img_pil.width - 5, 10 + text_h + 5)],
            fill="white",
        )
        draw.text((img_pil.width - text_w - 8, 12), text, fill="black", font=font)

        # Converter para bytes
        img_byte_arr = io.BytesIO()
        img_pil.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)

        return StreamingResponse(
            content=img_byte_arr,
            media_type="image/jpeg",
            headers={"X-Cisto-Count": str(cisto_count)},
        )

    elif content_type.startswith("video/"):
        # Verificar extensão válida
        if not file.filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
            return JSONResponse(
                content={"error": "Formato de vídeo não suportado"},
                status_code=400,
            )

        # Processar vídeo
        results = segmentation_model.predict(
            source=str(file_path),
            save=True, 
            show_labels=False,
            project=UPLOAD_DIR,
            name="processed_seg",
            vid_stride=1,
        )

        # Contar cistos totais no vídeo
        cisto_count = sum(len(r.boxes) for r in results)

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
        return JSONResponse(
            content={
                "video_path": str(output_video),
                "cisto_count": cisto_count,
            },
            status_code=200,
        )

    else:
        return JSONResponse(
            content={"error": "Tipo de arquivo não suportado"},
            status_code=400,
        )

@app.post("/predict/segmentation/batch")
async def predict_segmentation_batch(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    batch_dir = UPLOAD_DIR / f"batch_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    output_dir = batch_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_files = []

    for file in files:
        content_type = file.content_type
        filename = f"{datetime.now().strftime('%H%M%S')}_{file.filename}"
        file_path = batch_dir / filename

        with open(file_path, "wb") as f:
            f.write(await file.read())

        if content_type.startswith("image/"):
            results = segmentation_model.predict(source=str(file_path), save=False, imgsz=640, show_labels=False)
            segmented_img = results[0].plot()
            img_pil = Image.fromarray(segmented_img)

            output_file = output_dir / f"pred_{filename}.jpg"
            img_pil.save(output_file)
            processed_files.append(output_file)

        elif content_type.startswith("video/"):
            if not file.filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue

            results = segmentation_model.predict(
                source=str(file_path),
                save=True,
                show_labels=False,
                project=output_dir,
                name=f"video_{filename}",
                vid_stride=1,
            )

            save_dir = results[0].save_dir
            output_videos = list(Path(save_dir).rglob("*.*"))
            output_videos = [
                f for f in output_videos if f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]
            ]

            if output_videos:
                processed_files.extend(output_videos)

        else:
            # Tipo de arquivo não suportado
            continue

    if not processed_files:
        return JSONResponse(
            content={"error": "Nenhum arquivo válido processado."},
            status_code=400,
        )

    zip_path = batch_dir / f"predictions_{timestamp}.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for file in processed_files:
            zipf.write(file, arcname=file.name)

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=zip_path.name,
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

    results = classification_model.predict(source=str(file_path), save=False, imgsz=640, show_labels=False)

    probs = results[0].probs  
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
        show_labels=False,
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



#Outro
# 

import base64
from io import BytesIO
from fastapi.responses import JSONResponse
from fastapi import Request

@app.post("/predict/policistos")
async def predict_policistos(
    request: Request,
    files: List[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    batch_dir = UPLOAD_DIR / f"policistos_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    output_dir = batch_dir / "labeled"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for file in files:
        filename = f"{datetime.now().strftime('%H%M%S')}_{file.filename}"
        file_path = batch_dir / filename

        with open(file_path, "wb") as f:
            f.write(await file.read())

        if filename.endswith(".zip"):
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

        else:
            if file.content_type.startswith("image/"):
                processed = await process_single_image(file_path, output_dir)
                if processed:
                    results.append(processed)

    if not results:
        raise HTTPException(status_code=400, detail="Nenhuma imagem válida processada.")

    zip_path = batch_dir / f"results_{timestamp}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for r in results:
            zipf.write(r["path"], arcname=r["path"].name)

    base_url = str(request.base_url).rstrip("/")
    report_url = f"{base_url}/download/{zip_path.name}"

    response = {
        "results": [
            {
                "image": r["image_base64"],
                "info": {
                    "cistos": r["cistos"]
                }
            }
            for r in results
        ],
        "report_url": report_url
    }

    return JSONResponse(content=response)


async def process_single_image(img_path: Path, output_dir: Path):
    """Processa imagem e retorna dict com base64, cistos e path"""
    cls_results = classification_model.predict(
        source=str(img_path), save=False, imgsz=640, show_labels=False
    )
    probs = cls_results[0].probs
    if probs is None:
        return None

    top1_idx = int(probs.top1)
    class_name = classification_model.names[top1_idx].lower()

    if class_name == "saudavel":
        return None

    seg_results = segmentation_model.predict(
        source=str(img_path), save=False, imgsz=640, show_labels=False
    )
    cisto_count = len(seg_results[0].boxes)

    seg_img = seg_results[0].plot(labels=False)
    img_pil = Image.fromarray(seg_img).convert("RGB")

    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.load_default()
    text = f"Cistos: {cisto_count}"
    draw.text((10, 10), text, font=font, fill="red")

    output_file = output_dir / f"{img_path.stem}_cistos{cisto_count}.jpg"
    img_pil.save(output_file)

    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    img_base64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

    return {
        "path": output_file,
        "image_base64": img_base64,
        "cistos": cisto_count
    }


from fastapi.responses import FileResponse

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    return FileResponse(path=file_path, filename=filename, media_type="application/zip")