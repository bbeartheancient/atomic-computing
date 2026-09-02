"""atomic.ui — ATOMIC-PC tile wall UI.

Exposes the FastAPI app, the Viewer session manager, and a uvicorn
runner.  The UI is a pure-web client (HTML5 canvas per tile) that
connects to the server via WebSocket or REST for live tick streaming.

Run:
    python -m atomic.ui [--port 18094]
    uvicorn atomic.ui:app --port 18094 --host 0.0.0.0
"""
from .server import app
from .viewer import Viewer
from .programs import build, all_programs

__all__ = ["app", "Viewer", "build", "all_programs"]


def run_server(host: str = "0.0.0.0", port: int = 18094):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import sys
    port = 18094
    for a in sys.argv[1:]:
        if a.isdigit():
            port = int(a)
    run_server(port=port)
