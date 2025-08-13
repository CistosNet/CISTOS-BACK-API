import cv2
from ultralytics import YOLO
import numpy as np
import os
import glob


modelo = YOLO("modelo_yolo_teste/best.pt")

def detectar_objetos_pasta(model, pasta, conf=0.2):
    pasta_resultados = os.path.join("resultados")
    os.makedirs(pasta_resultados, exist_ok=True)

    imagens = glob.glob(os.path.join(pasta, "*.jpg")) + glob.glob(os.path.join(pasta, "*.png"))

    for img_path in imagens:
        results = model.predict(source=img_path, conf=conf)

        for r in results:
            img = r.orig_img.copy()

            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            masks = r.masks.data.cpu().numpy() if r.masks is not None else []

            for box, conf_score, cls, mask in zip(boxes, confs, classes, masks):
                label_name = model.names[int(cls)]
                print(f"[{os.path.basename(img_path)}] - {label_name}: {conf_score:.4f}")

                mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)

                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{label_name} {conf_score:.2f}"
                cv2.putText(img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        nome_saida = os.path.join(pasta_resultados, os.path.basename(img_path))
        cv2.imwrite(nome_saida, img)

    print("✅ Processamento concluído!")

detectar_objetos_pasta(modelo, r"base_pequena")
