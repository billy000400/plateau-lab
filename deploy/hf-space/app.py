"""Entry point for a Hugging Face Gradio Space.

The Gradio SDK installs requirements.txt and runs `python app.py`; the Space serves
whatever listens on port 7860. This starts the FastAPI app (API + web/ frontend)
there. Gradio itself is installed by the SDK but not used.
"""
import os

try:
    import spaces  # preinstalled on ZeroGPU Spaces; must be imported before torch

    # ZeroGPU refuses to start without at least one @spaces.GPU function. This stub is
    # never called: inference runs on NDIF, so no GPU (or GPU quota) is used.
    @spaces.GPU
    def _zerogpu_stub():
        pass
except ImportError:  # not on a Hugging Face Space
    spaces = None

os.environ.setdefault("PLATEAU_REMOTE", "1")  # NDIF with users' own keys or the lab access code

import uvicorn

from server.app import app

if __name__ == "__main__":
    # ZeroGPU reports the GPU functions from inside gradio.Blocks.launch(). We serve with
    # uvicorn instead, so run that startup step directly (only defined on ZeroGPU).
    zerogpu_startup = getattr(getattr(spaces, "zero", None), "startup", None)
    if zerogpu_startup is not None:
        zerogpu_startup()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
