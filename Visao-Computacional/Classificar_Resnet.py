import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# --------------------------
# 1. Carregar o modelo binário
# --------------------------
model = load_model(r"C:\Users\joaol\Empresa\best_resnet50.h5")

# Definir nomes das classes
class_names = ["Classe_0", "Classe_1"]  # Exemplo: ["Normal", "Cisto"]

# --------------------------
# 2. Função para classificar imagem usando PIL
# --------------------------
def classificar_imagem(model, img_path):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Erro ao abrir a imagem: {e}")
        return None

    # Redimensionar para 224x224
    img_resized = img.resize((224, 224))
    x = np.array(img_resized) / 255.0  # Normalização
    x = np.expand_dims(x, axis=0)

    # Predição
    pred = model.predict(x)[0][0]

    # Determinar classe e probabilidades
    probas = [1 - pred, pred]
    label = class_names[1] if pred >= 0.5 else class_names[0]

    for i, p in enumerate(probas):
        print(f"{class_names[i]}: {p:.4f}")

    # Converter de volta para NumPy para visualização (opcional)
    img_np = np.array(img)
    return img_np, label, probas

# --------------------------
# 3. Testar
# --------------------------
imagem = r"Visao-Computacional\Imagem do WhatsApp de 2025-09-01 à(s) 23.02.44_bb243d08.jpg"
img_np, label, probas = classificar_imagem(model, imagem)
print(f"Classe prevista: {label}")
