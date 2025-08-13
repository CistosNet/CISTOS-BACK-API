import cv2
import os

caminho = r"base_teste"

imagens = sorted([img for img in os.listdir(caminho) if img.endswith((".png", ".jpg", ".jpeg"))])

frame = cv2.imread(os.path.join(caminho, imagens[0]))
altura, largura, _ = frame.shape

saida = cv2.VideoWriter('video_saida.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (largura, altura))

for img_nome in imagens:
    img = cv2.imread(os.path.join(caminho, img_nome))
    saida.write(img)

saida.release()
print("Vídeo MP4 gerado com sucesso!")
