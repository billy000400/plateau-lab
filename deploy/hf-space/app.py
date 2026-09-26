"""Entry point for a Hugging Face Gradio Space.

The Gradio SDK installs requirements.txt and runs `python app.py`; the Space serves
whatever listens on port 7860. This starts the FastAPI app (API + web/ frontend)
there. Gradio itself is installed by the SDK but not used.
"""
import os

os.environ.setdefault("PLATEAU_REMOTE", "1")  # NDIF with users' own keys or the lab access code

import uvicorn

from server.app import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
