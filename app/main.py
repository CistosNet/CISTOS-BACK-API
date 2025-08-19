from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

model_path = Path(__file__).resolve().parent.parent / "Visao-Computacional" / "modelo_yolo_teste" / "best.pt"
model = YOLO(str(model_path))

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def read_root():
    return {"message": "rota 01"}


@app.get("/coffee")
def coffee():
    return JSONResponse(
        status_code=418,
        content={"message": "I'm a Teapot"}
    )


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if file is None:
        return JSONResponse(content={"error": "Nenhum arquivo recebido"}, status_code=400)

    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    file_path = UPLOAD_DIR / filename

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    results = model.predict(source=str(file_path), save=False, imgsz=640)

    segmented_img = results[0].plot()

    img_pil = Image.fromarray(segmented_img)
    img_byte_arr = io.BytesIO()
    img_pil.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    return StreamingResponse(content=img_byte_arr, media_type="image/jpeg")
