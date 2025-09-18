import cv2
from ultralytics import YOLO
import numpy as np

# Carregar o modelo de classificação YOLO
model = YOLO("modelos_yolo_2/best_classificador.pt")

def classificar_imagem(model, img_path, conf=0.2):
    results = model.predict(source=img_path, conf=conf)

    for r in results:
        img = r.orig_img.copy()

        if r.probs is None or len(r.probs) == 0:
            print("Nenhuma classe detectada na imagem.")
            continue

        probs = r.probs[0].cpu().numpy() 
        classes = r.names 

        for idx, prob in enumerate(probs):
            label_name = classes[idx]
            print(f" - {label_name}: {prob:.4f}")

            label = f"{label_name} {prob:.2f}"
            cv2.putText(
                img, label, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

    _, img_encoded = cv2.imencode(".jpg", img)
    return img_encoded.tobytes()

imagem = r"C:\Users\joaol\OneDrive\Imagens\Imagem do WhatsApp de 2025-09-01 à(s) 23.02.44_bb243d08.jpg"
classificar_imagem(model, imagem)
