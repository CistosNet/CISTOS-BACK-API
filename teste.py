import os

IGNORAR = {'.git'}  # Pastas ou arquivos a ignorar

def mostrar_estrutura(pasta, prefixo=""):
    itens = [item for item in os.listdir(pasta) if item not in IGNORAR]
    for i, item in enumerate(itens):
        caminho_completo = os.path.join(pasta, item)
        conector = "├── " if i < len(itens) - 1 else "└── "
        print(prefixo + conector + item)
        if os.path.isdir(caminho_completo):
            novo_prefixo = prefixo + ("│   " if i < len(itens) - 1 else "    ")
            mostrar_estrutura(caminho_completo, novo_prefixo)

# Altere para o caminho da sua pasta do projeto
caminho_projeto = r"C:\Users\joaol\Empresa\CISTOS_EXECUTAVEL"
print(caminho_projeto)
mostrar_estrutura(caminho_projeto)
