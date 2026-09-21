import asyncio
import logging
import os
import sys
import time
from collections import deque

from backend.data import router as data_router
from backend.evaluation import router as evaluation_router
from backend.hpo import router as hpo_router
from backend.models import router as model_router
from backend.preprocess import router as preprocess_router
from backend.sam import router as sam_router
from backend.temporal_trainers import router as temporal_trainers_router
from backend.trainers import router as trainers_router
from backend.vae_trainers import router as vae_trainers_router
from backend.visualization import router as visualization_router
from backend.workflows import router as workflows_router
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

logs_router = APIRouter()

_LOG_BUFFER = deque(maxlen=2000)
_LOG_QUEUE: asyncio.Queue[str] = asyncio.Queue(maxsize=2000)


def _push_line(line: str) -> None:
    line = (line or "").rstrip("\n")
    if not line:
        return
    _LOG_BUFFER.append(line)
    try:
        _LOG_QUEUE.put_nowait(line)
    except asyncio.QueueFull:
        # drop if overloaded
        pass


class _QueueLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        _push_line(msg)


class _TeeStream:
    """Mirror stdout/stderr into the log queue while keeping normal terminal output."""

    def __init__(self, stream, prefix=""):
        self._stream = stream
        self._prefix = prefix

    def write(self, s):
        self._stream.write(s)
        self._stream.flush()
        for line in str(s).splitlines():
            _push_line(self._prefix + line)

    def flush(self):
        self._stream.flush()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", None)

    def isatty(self):
        return bool(getattr(self._stream, "isatty", lambda: False)())

    def __getattr__(self, name):
        return getattr(self._stream, name)


def install_terminal_mirror():
    # Guard against double-install (uvicorn reload can import twice)
    if getattr(install_terminal_mirror, "_installed", False):
        return
    install_terminal_mirror._installed = True

    handler = _QueueLogHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    # Capture Python logging (including uvicorn loggers)
    for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.addHandler(handler)
        if lg.level == logging.NOTSET:
            lg.setLevel(logging.INFO)

    # Capture print() and anything writing to stdout/stderr
    sys.stdout = _TeeStream(sys.stdout)
    sys.stderr = _TeeStream(sys.stderr, prefix="STDERR: ")


@logs_router.get("/logs/stream")
async def logs_stream():
    async def gen():
        # Send a little backlog first
        for line in list(_LOG_BUFFER)[-200:]:
            yield f"data: {line}\n\n"

        # Then stream live
        while True:
            line = await _LOG_QUEUE.get()
            yield f"data: {line}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


class NoCacheNodesMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)

        if request.url.path.startswith("/nodes/") or request.url.path.startswith(
            "/static/workflows/"
        ):
            response.headers["Cache-Control"] = (
                "no-cache, no-store, must-revalidate, max-age=0"
            )
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

        return response


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
NODES_DIR = os.path.join(BASE_DIR, "melt_nodes")
DEFAULT_HOST = os.environ.get("MELT_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("MELT_PORT", "8000"))

app = FastAPI()

install_terminal_mirror()
app.include_router(logs_router)

# Allow calls from your frontend host (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(NoCacheNodesMiddleware)


# The frontend are static files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/nodes", StaticFiles(directory=NODES_DIR), name="nodes")

app.include_router(preprocess_router)
app.include_router(data_router)
app.include_router(visualization_router)
app.include_router(trainers_router)
app.include_router(temporal_trainers_router)
app.include_router(vae_trainers_router)
app.include_router(evaluation_router)
app.include_router(model_router)
app.include_router(workflows_router)
app.include_router(sam_router)
app.include_router(hpo_router)


@app.on_event("startup")
async def announce_startup():
    print(f"MELT-UI running at http://{DEFAULT_HOST}:{DEFAULT_PORT}/")


@app.get("/health")
def health():
    return {
        "ok": True,
        "app": "melt-ui",
        "url": f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/",
    }


# Launch the frontend
@app.get("/")
def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
