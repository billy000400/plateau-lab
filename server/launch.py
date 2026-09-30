"""Compatible local launcher for the FastAPI application."""
import argparse
import os
from pathlib import Path
import threading
import webbrowser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    # The local launcher is always local, even if a shell has an NDIF key set.
    os.environ["PLATEAU_REMOTE"] = "0"
    if args.data_dir:
        os.environ["PLATEAU_DATA_DIR"] = str(args.data_dir.resolve())
    import uvicorn
    if args.open:
        threading.Timer(1, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}/")).start()
    uvicorn.run("server.app:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
