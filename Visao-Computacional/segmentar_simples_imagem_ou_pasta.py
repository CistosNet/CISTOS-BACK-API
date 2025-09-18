from ultralytics import YOLO
import os

model = YOLO("modelo_yolo_teste/best.pt")
resultados_pasta = "resultados"
os.makedirs(resultados_pasta, exist_ok=True)

def predizer_segmentacao(path):
    

    results = model.predict(path, conf=0.18, classes=[1], save_conf=True)
    caminhos_salvos = []

    for r in results:
        nome_arquivo = os.path.basename(r.path)
        nome_base, ext = os.path.splitext(nome_arquivo)
        nome_arquivo_predict = f"{nome_base}_predict{ext}"
        caminho_salvar = os.path.join(resultados_pasta, nome_arquivo_predict)
        r.save(caminho_salvar)
        caminhos_salvos.append(caminho_salvar)


#predizer_segmentacao(r"base_pequena") 
predizer_segmentacao(r"base_teste/Tr-pi_1114_jpg.rf.d7932c9af74196f293163c2178684cd6.jpg") 