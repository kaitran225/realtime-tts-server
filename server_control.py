"""Engine supervisor and control API for the Sylvia TTS server.

Owns the Qwen3-TTS engine process (start / stop / restart), exposes the model
catalog and accepts configuration pushed by the client.

    python server_control.py
    Swagger: http://127.0.0.1:2557/docs
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Body, FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / ".config"
MODELS_PATH = CONFIG_DIR / "models.json"
SETTINGS_PATH = CONFIG_DIR / "control.json"
CLIENT_CONFIG_PATH = CONFIG_DIR / "client-config.json"
LOG_DIR = ROOT / "logs"

ENGINE_PORT = 2558
LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 2557
BACKENDS = {
    "cpu": ".venv",
    "gpu": ".venv-gpu",
}
START_TIMEOUT = 180.0
STOP_TIMEOUT = 30.0

LOG_DIR.mkdir(parents=True, exist_ok=True)
LOGGER = logging.getLogger("sylvia.control")
LOGGER.setLevel(logging.DEBUG)
LOGGER.propagate = False
_handler = logging.handlers.RotatingFileHandler(
    LOG_DIR / f"control-{datetime.now():%Y%m%d}.log",
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
LOGGER.addHandler(_handler)
_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
LOGGER.addHandler(_console)

_lock = threading.Lock()
_child: subprocess.Popen[Any] | None = None
_child_log: Path | None = None
_started_at: float | None = None
NOWINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# ------------------------------------------------------------------------ helpers


def _read_json(path: Path, default: Any) -> Any:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp.replace(path)


def catalog() -> dict:
    return _read_json(MODELS_PATH, {})


def settings() -> dict:
    stored = _read_json(SETTINGS_PATH, {})
    models = catalog()
    defaults = models.get("defaults") or {}
    device = stored.get("device")
    if device not in BACKENDS:
        device = "gpu" if (ROOT / BACKENDS["gpu"] / "Scripts" / "python.exe").is_file() else "cpu"
    return {
        "device": device,
        "model": stored.get("model") or defaults.get("serve") or "0.6b-base-q8",
        "design_model": stored.get("design_model") or defaults.get("design") or "1.7b-voicedesign-q4",
        "new_reference": bool(stored.get("new_reference", False)),
    }


def _talker(model_id: str) -> dict | None:
    for talker in catalog().get("talkers") or []:
        if talker.get("id") == model_id:
            return talker
    return None


def engine_ready(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{ENGINE_PORT}/ready", timeout=timeout
        ) as response:
            return json.loads(response.read().decode("utf-8")).get("status") == "ready"
    except Exception:
        return False


def _engine_python(device: str) -> Path:
    venv = BACKENDS.get(device)
    if venv is None:
        raise ValueError(f"unknown device {device!r}; use cpu or gpu")
    python = ROOT / venv / "Scripts" / "python.exe"
    if not python.is_file():
        raise FileNotFoundError(f"{venv} is not installed; run Setup-TTS{'GPU' if device == 'gpu' else ''}.bat")
    return python


def _status() -> dict:
    with _lock:
        running = _child is not None and _child.poll() is None
        pid = _child.pid if running and _child else None
        external = (not running) and engine_ready()
        payload = {
            "ok": True,
            "running": running or external,
            "managed": running,
            "external": external,
            "pid": pid,
            "engine_ready": engine_ready(),
            "uptime_s": round(time.monotonic() - _started_at, 1) if running and _started_at else None,
            "log_file": str(_child_log) if _child_log else None,
            "engine_url": f"http://127.0.0.1:{ENGINE_PORT}",
            "studio_url": f"http://127.0.0.1:{ENGINE_PORT}/studio",
        }
        payload.update(settings())
        return payload


def _tail(path: Path | None, lines: int = 40) -> list[str]:
    if path is None or not path.is_file():
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read().splitlines()[-lines:]
    except OSError:
        return []


# ------------------------------------------------------------------ process control


def _stop() -> dict:
    global _child, _started_at
    with _lock:
        child = _child
        _child = None
        _started_at = None
    if child is None or child.poll() is not None:
        return {"stopped": False, "reason": "no managed engine process"}
    LOGGER.info("stopping engine pid=%s", child.pid)
    broke = True
    try:
        child.send_signal(signal.CTRL_BREAK_EVENT)
    except Exception as exc:
        broke = False
        LOGGER.warning("CTRL_BREAK failed (%s); killing the process tree", exc)
    try:
        if not broke:
            raise subprocess.TimeoutExpired(str(child.pid), 0)
        child.wait(timeout=STOP_TIMEOUT)
        LOGGER.info("engine stopped cleanly")
        return {"stopped": True, "clean": True}
    except subprocess.TimeoutExpired:
        LOGGER.warning("engine did not stop in %.0fs; killing tree", STOP_TIMEOUT)
        subprocess.run(
            ["taskkill", "/PID", str(child.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            creationflags=NOWINDOW,
        )
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
        return {"stopped": True, "clean": False}


def _start(wait: bool = True) -> dict:
    global _child, _child_log, _started_at
    config = settings()
    python = _engine_python(config["device"])
    for key, role in (("model", "serve"), ("design_model", "design")):
        talker = _talker(config[key])
        if talker is None:
            raise ValueError(f"{key} {config[key]!r} is not in .config/models.json")
        if role not in (talker.get("roles") or []):
            raise ValueError(f"{key} {config[key]!r} is not usable for {role}")

    if engine_ready():
        LOGGER.warning("engine already answering on %s; refusing to double-start", ENGINE_PORT)
        return {"started": False, "reason": "port busy"}

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"engine-{datetime.now():%Y%m%d-%H%M%S}.log"
    command = [
        str(python),
        str(ROOT / "start_sylvia_server.py"),
        "--device", config["device"],
        "--model", config["model"],
        "--design-model", config["design_model"],
        "--yes",
    ]
    if config["new_reference"]:
        command.append("-new")
    LOGGER.info("starting engine: %s", " ".join(command))
    handle = open(log_path, "w", encoding="utf-8", buffering=1)
    child = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | NOWINDOW,
    )
    with _lock:
        _child = child
        _child_log = log_path
        _started_at = time.monotonic()

    if not wait:
        return {"started": True, "pid": child.pid, "log_file": str(log_path), "ready": False}

    deadline = time.monotonic() + START_TIMEOUT
    while time.monotonic() < deadline:
        if child.poll() is not None:
            LOGGER.error("engine exited during startup (code %s)", child.returncode)
            return {
                "started": False,
                "error": f"engine exited with code {child.returncode}",
                "log_file": str(log_path),
                "tail": _tail(log_path, 25),
            }
        if engine_ready(1.5):
            LOGGER.info("engine ready (pid=%s)", child.pid)
            return {
                "started": True,
                "ready": True,
                "pid": child.pid,
                "log_file": str(log_path),
                "waited_s": round(START_TIMEOUT - (deadline - time.monotonic()), 1),
            }
        time.sleep(2)
    LOGGER.error("engine not ready within %.0fs", START_TIMEOUT)
    return {"started": True, "ready": False, "pid": child.pid, "log_file": str(log_path)}


def _restart(patch: dict | None = None) -> dict:
    if patch:
        merged = settings()
        merged.update({k: v for k, v in patch.items() if v not in (None, "")})
        _write_json(SETTINGS_PATH, merged)
        LOGGER.info("settings updated: %s", merged)
    stop_result = _stop()
    time.sleep(1.5)
    try:
        start_result = _start(wait=True)
    except Exception as exc:
        LOGGER.error("restart failed: %s", exc)
        return {"ok": False, "error": str(exc), "stop": stop_result}
    return {"ok": bool(start_result.get("ready") or start_result.get("started")),
            "stop": stop_result, "start": start_result, "status": _status()}


# --------------------------------------------------------------------------- API


class RestartRequest(BaseModel):
    device: str | None = None
    model: str | None = None
    design_model: str | None = None
    new_reference: bool | None = None


class ClientConfig(BaseModel):
    config: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    LOGGER.info("control API starting on %s:%s", LISTEN_HOST, LISTEN_PORT)
    if engine_ready():
        LOGGER.info("engine already running (external); adopt without restart")
    else:
        threading.Thread(target=lambda: _start(wait=True), daemon=True).start()
    yield
    LOGGER.info("control API stopping")
    _stop()


app = FastAPI(
    title="Sylvia TTS Server Control",
    version="1.0",
    description=(
        "Supervises the Qwen3-TTS engine: status, model catalog, client configuration "
        "intake, and on-demand restart. Swagger at /docs."
    ),
    lifespan=lifespan,
)


@app.get("/health", tags=["status"], summary="Control API liveness")
def health() -> dict:
    return {"status": "ok", "engine_ready": engine_ready()}


@app.get("/api/status", tags=["status"], summary="Engine status, settings and log path")
def status() -> dict:
    payload = _status()
    payload["tail"] = _tail(_child_log, 20)
    return payload


@app.get("/api/settings", tags=["config"], summary="Engine settings used on next start")
def get_settings() -> dict:
    return settings()


@app.put("/api/settings", tags=["config"], summary="Persist engine settings")
def put_settings(patch: dict = Body(...)) -> dict:
    merged = settings()
    merged.update({k: v for k, v in (patch or {}).items() if v not in (None, "")})
    _write_json(SETTINGS_PATH, merged)
    LOGGER.info("settings saved: %s", merged)
    return {"ok": True, "settings": merged}


@app.get("/api/catalog", tags=["config"], summary="Model catalog (models.json)")
def get_catalog() -> dict:
    return catalog()


@app.put("/api/catalog", tags=["config"], summary="Replace the model catalog")
def put_catalog(payload: dict = Body(...)) -> dict:
    if not isinstance(payload.get("talkers"), list):
        return {"ok": False, "error": "catalog needs a 'talkers' list"}
    _write_json(MODELS_PATH, payload)
    LOGGER.info("catalog saved (%s talkers)", len(payload["talkers"]))
    return {"ok": True, "catalog": catalog()}


@app.post("/api/restart", tags=["engine"], summary="Restart the engine, optionally with new models")
def restart(payload: RestartRequest = Body(default_factory=RestartRequest)) -> dict:
    patch = payload.model_dump(exclude_none=True)
    return _restart(patch)


@app.post("/api/start", tags=["engine"], summary="Start the engine if it is down")
def start() -> dict:
    if engine_ready():
        return {"ok": True, "started": False, "reason": "already running", "status": _status()}
    result = _start(wait=True)
    return {"ok": bool(result.get("ready") or result.get("started")), "start": result, "status": _status()}


@app.post("/api/stop", tags=["engine"], summary="Stop the managed engine process")
def stop() -> dict:
    return {"ok": True, "stop": _stop(), "status": _status()}


@app.post("/api/client-config", tags=["config"], summary="Accept configuration pushed by the client")
def client_config(payload: ClientConfig) -> dict:
    config = payload.config or {}
    _write_json(CLIENT_CONFIG_PATH, {"received_at": datetime.now().isoformat(timespec="seconds"),
                                     "config": config})
    LOGGER.info(
        "client config received: keys=%s volume=%s",
        sorted(config),
        (config.get("audio") or {}).get("volume_db"),
    )
    return {
        "ok": True,
        "stored": str(CLIENT_CONFIG_PATH),
        "keys": sorted(config),
        "settings": settings(),
    }


@app.get("/api/client-config", tags=["config"], summary="Last client configuration received")
def get_client_config() -> dict:
    return _read_json(CLIENT_CONFIG_PATH, {"config": None})


@app.get("/api/logs/tail", tags=["status"], summary="Tail the current engine log")
def logs_tail(lines: int = 40) -> dict:
    return {"log_file": str(_child_log) if _child_log else None, "lines": _tail(_child_log, max(1, min(lines, 500)))}


if __name__ == "__main__":
    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT, log_level="info", access_log=False)
