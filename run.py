import threading
import time
import webbrowser
import uvicorn
from app.main import app

def start_server():
    uvicorn.run(app, host="127.0.0.1", port=5000, reload=False, log_config=None)

if __name__ == "__main__":
    # roda o servidor em uma thread
    threading.Thread(target=start_server, daemon=True).start()
    # abre o navegador automaticamente
    webbrowser.open("http://127.0.0.1:5000/index.html")
    # mantém o programa vivo
    while True:
        time.sleep(1)
