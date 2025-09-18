# 🚀 CistoNet API
API feita com FastAPI para processamento e resposta de imagens.

## 📁 Estrutura do Projeto
```csharp
CistoNet-API/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── routes/
│   └── models/
├── requirements.txt
├── .gitignore
└── .venv/  ← ambiente virtual (ignorado pelo git)
```

## 🧑‍💻 Pré-requisitos

 . Python 3.12 instalado

 . Git

 . Virtualenv (opcional, mas recomendado)

## 📦 Instalação e Execução
### 1. Clone o projeto
```bash
    git clone https://github.com/seu-usuario/CistoNet-API.git
    cd CistoNet-API
```

### 2. Crie e ative o ambiente virtual
```bash
    python3 -m venv .venv
    source .venv/bin/activate  # Linux/macOS
```

### 3. Instale as dependências
```bash
    pip install -r requirements.txt
```


### 4. Rode o servidor
```bash
    uvicorn app.main:app --reload
```


## 📫 Endpoints e Documentação

Swagger UI: http://localhost:8000/docs

ReDoc: http://localhost:8000/redoc

## 🧹 Limpeza
Para apagar caches e arquivos temporários:
```bash
find . -type d -name "__pycache__" -exec rm -r {} +
```
