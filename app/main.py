from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
# import os
from datetime import datetime
from pathlib import Path

# from fastapi import FastAPI, File, UploadFile
#from fastapi.responses import StreamingResponse
from ultralytics import YOLO
#from PIL import Image
#import io

app = FastAPI()

model = YOLO("../Visao-computacional/modelo_yolo_teste/best.pt")

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
    print("📦 file recebido:", file)
    if file is None:
        return JSONResponse(content={
            "error": "Nenhum arquivo recebido"
            }, status_code=400)

    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    file_path = UPLOAD_DIR / filename

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return JSONResponse(
        status_code=201,
        content={"message": "Imagem salva com sucesso!", "file_path": str(file_path)}
    )
