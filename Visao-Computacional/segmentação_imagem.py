import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("modelos_yolo/best_segmentador.pt")

def detectar_objetos_imagem(model, img_path, conf=0.2):
    results = model.predict(source=img_path, conf=conf)

    for r in results:
        img = r.orig_img.copy()

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        masks = r.masks.data.cpu().numpy() if r.masks is not None else []

        #print(f"\n📄 Resultados para: {img_path}")

        for box, conf_score, cls, mask in zip(boxes, confs, classes, masks):
            label_name = model.names[int(cls)]
            print(f" - {label_name}: {conf_score:.4f}")

            mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
            mask_binary = (mask_resized > 0.5).astype(np.uint8)

            contours, _ = cv2.findContours(
                mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{label_name} {conf_score:.2f}"
            cv2.putText(
                img, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

    _, img_encoded = cv2.imencode(".jpg", img)
    return img_encoded.tobytes()


imagem = r"Visao-Computacional\base_teste\img_92_jpg.rf.5e2862d76dd7376fc4df0838c1ccf34d.jpg"
detectar_objetos_imagem(model, imagem)

