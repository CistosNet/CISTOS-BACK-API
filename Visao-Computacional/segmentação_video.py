import cv2
import numpy as np
import os
from ultralytics import YOLO

def detectar_objetos_video(model, video_path, conf=0.2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Erro ao abrir o vídeo: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pasta_resultados = os.path.join(os.path.dirname(video_path), "resultados")
    pasta_frames = os.path.join(pasta_resultados, "frames")
    os.makedirs(pasta_resultados, exist_ok=True)
    os.makedirs(pasta_frames, exist_ok=True)

    saida_video = os.path.join(pasta_resultados, "resultado_video.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(saida_video, fourcc, fps, (largura, altura))

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=conf, verbose=False)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            masks = r.masks.data.cpu().numpy() if r.masks is not None else []

            for box, conf_score, cls, mask in zip(boxes, confs, classes, masks):
                label_name = model.names[int(cls)]

                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{label_name} {conf_score:.2f}"
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        frame_path = os.path.join(pasta_frames, f"frame_{frame_id:05d}.jpg")
        cv2.imwrite(frame_path, frame)

        out.write(frame)

        frame_id += 1

    cap.release()
    out.release()

    print(f"✅ Vídeo processado salvo em: {saida_video}")
    print(f"🖼️ Frames salvos em: {pasta_frames}")


modelo = YOLO("modelo_yolo_teste/best.pt")
detectar_objetos_video(modelo, r"tomografia.mp4")
