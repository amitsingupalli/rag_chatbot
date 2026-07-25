from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["PYTHONPATH"] = str(ROOT)


def main() -> None:
    print("Starting RAG Chatbot Application...")
    backend = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )

    time.sleep(2)

    frontend = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT / "frontend" / "app.py"),
            "--server.port",
            "8501",
            "--server.address",
            "0.0.0.0",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )

    print("Backend running at http://0.0.0.0:8000")
    print("Frontend running at http://0.0.0.0:8501")

    try:
        frontend.wait()
    except KeyboardInterrupt:
        pass
    finally:
        backend.terminate()
        frontend.terminate()


if __name__ == "__main__":
    main()
