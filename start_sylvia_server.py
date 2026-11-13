"""Generate a Sylvia reference with VoiceDesign, then serve it from a Base talker.

GGUF paths live in .config/models.json. The launcher prints the available talkers
and asks which one to serve, unless --model/--design-model or --yes is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import logging.handlers
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / ".config"
MODELS_CONFIG_PATH = CONFIG_DIR / "models.json"
VOICE_PROFILE_PATH = CONFIG_DIR / "voice-profile.txt"
SEED_SCRIPT_PATH = CONFIG_DIR / "sylvia-seed-script.txt"
VOICE_DIR = ROOT / "voices"
REFERENCE_DIR = ROOT / "reference"
ARCHIVE_REFERENCE_DIR = ROOT / "archive-reference"
REFERENCE_PATH = REFERENCE_DIR / "sylvia_reference.wav"
REFERENCE_TEXT_PATH = REFERENCE_DIR / "sylvia_reference.txt"
DESIGN_PORT = 2559
SERVER_PORT = 2558
VOICE_NAME = "sylvia"
SERVER_ROLE = "serve"
DESIGN_ROLE = "design"
VALID_MODEL_TYPES = ("base", "voice_design", "custom_voice")
LOG_DIR = ROOT / "logs"
LOGGER = logging.getLogger("sylvia.server")

# Device-independent server flags; device-specific ones come from
# .config/models.json -> device_profiles[device].flags
COMMON_FLAGS = (
    "--max-active-requests", "1",
    "--max-queued-requests", "0",
    "--synthesis-timeout", "1800",
    "--lang", "English",
)


@dataclass(frozen=True)
class Talker:
    """One catalog entry: a talker GGUF plus the codec GGUF it ships with."""

    id: str
    label: str
    model_type: str
    quant: str
    alias: str
    path: Path
    codec_quant: str
    codec_path: Path
    roles: tuple[str, ...]
    note: str
    optional: bool

    @property
    def missing(self) -> tuple[Path, ...]:
        return tuple(path for path in (self.path, self.codec_path) if not path.is_file())

    @property
    def size_mb(self) -> float:
        try:
            return self.path.stat().st_size / 1_048_576
        except OSError:
            return 0.0


@dataclass(frozen=True)
class Catalog:
    root: Path
    talkers: dict[str, Talker]
    defaults: dict[str, str]
    codecs: dict[str, Path]
    device_profiles: dict[str, tuple[str, ...]]
    default_device: str

    def for_role(self, role: str, *, available_only: bool = True) -> list[Talker]:
        models = [model for model in self.talkers.values() if role in model.roles]
        return [model for model in models if not model.missing] if available_only else models

    def default_for(self, role: str) -> Talker | None:
        wanted = self.defaults.get(role)
        available = self.for_role(role)
        match = next((model for model in available if model.id == wanted), None)
        return match if match is not None else (available[0] if available else None)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and serve the Sylvia voice.",
        epilog="Model paths are read from .config/models.json; use --list to inspect them.",
    )
    parser.add_argument(
        "--model",
        dest="serve_model",
        metavar="ID",
        help=f"talker id to serve (see --list); skips the interactive picker",
    )
    parser.add_argument(
        "--design-model",
        dest="design_model",
        metavar="ID",
        help="VoiceDesign talker id used to build a new reference (see --list)",
    )
    parser.add_argument(
        "--list",
        dest="list_models",
        action="store_true",
        help="print the GGUF catalog with availability and exit",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="native runtime to use (default: auto -> gpu when the CUDA wheel is installed)",
    )
    parser.add_argument(
        "--log-level",
        choices=("debug", "info", "warning", "error"),
        default="info",
        help="console verbosity; the log file always records debug",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="resolve models and print the server commands without starting anything",
    )
    parser.add_argument(
        "-y",
        "--yes",
        dest="assume_yes",
        action="store_true",
        help="accept the configured default models instead of prompting",
    )
    parser.add_argument(
        "-new",
        "--new",
        dest="new_reference",
        action="store_true",
        help="archive the current reference and generate a new one",
    )
    return parser.parse_args(argv)


def _load_catalog() -> Catalog:
    if not MODELS_CONFIG_PATH.is_file():
        raise FileNotFoundError(MODELS_CONFIG_PATH)
    try:
        raw = json.loads(MODELS_CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{MODELS_CONFIG_PATH} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{MODELS_CONFIG_PATH}: top level must be a JSON object")

    models_root = raw.get("models_root")
    if not isinstance(models_root, str) or not models_root.strip():
        raise ValueError(f"{MODELS_CONFIG_PATH}: 'models_root' must be a non-empty string")
    root = Path(models_root.strip()).expanduser()
    if not root.is_absolute():
        root = (ROOT / root).resolve()

    codecs: dict[str, Path] = {}
    for quant, entry in (raw.get("codecs") or {}).items():
        name = entry.get("file") if isinstance(entry, dict) else entry
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{MODELS_CONFIG_PATH}: codec {quant!r} needs a 'file'")
        codecs[str(quant)] = root / name.strip()

    entries = raw.get("talkers")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{MODELS_CONFIG_PATH}: 'talkers' must be a non-empty list")

    talkers: dict[str, Talker] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{MODELS_CONFIG_PATH}: every talker entry must be an object")
        for key in ("id", "file", "roles"):
            if not entry.get(key):
                raise ValueError(f"{MODELS_CONFIG_PATH}: talker {entry!r} is missing {key!r}")
        ident = str(entry["id"]).strip()
        if ident in talkers:
            raise ValueError(f"{MODELS_CONFIG_PATH}: duplicate talker id {ident!r}")
        model_type = str(entry.get("model_type") or "base")
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(
                f"{MODELS_CONFIG_PATH}: talker {ident!r} has unknown model_type {model_type!r}"
            )
        roles = tuple(str(role).strip() for role in entry["roles"])
        unknown = [role for role in roles if role not in (SERVER_ROLE, DESIGN_ROLE)]
        if unknown:
            raise ValueError(f"{MODELS_CONFIG_PATH}: talker {ident!r} declares unknown roles {unknown}")
        quant = str(entry.get("quant") or "Q4_K_M")
        codec_quant = str(entry.get("codec") or quant)
        codec_path = codecs.get(codec_quant)
        if codec_path is None:
            raise ValueError(
                f"{MODELS_CONFIG_PATH}: talker {ident!r} uses undefined codec {codec_quant!r}"
            )
        talkers[ident] = Talker(
            id=ident,
            label=str(entry.get("label") or ident),
            model_type=model_type,
            quant=quant,
            alias=str(entry.get("alias") or f"qwen3-tts-{ident}-sylvia"),
            path=root / str(entry["file"]).strip(),
            codec_quant=codec_quant,
            codec_path=codec_path,
            roles=roles,
            note=str(entry.get("note") or ""),
            optional=bool(entry.get("optional", False)),
        )

    raw_defaults = raw.get("defaults") or {}
    defaults = {role: str(raw_defaults.get(role) or "") for role in (SERVER_ROLE, DESIGN_ROLE)}
    for role, ident in defaults.items():
        if ident and ident not in talkers:
            raise ValueError(
                f"{MODELS_CONFIG_PATH}: default {role} model {ident!r} is not a catalog entry"
            )

    device_profiles: dict[str, tuple[str, ...]] = {}
    for device, entry in (raw.get("device_profiles") or {}).items():
        flags = entry.get("flags") if isinstance(entry, dict) else entry
        if not isinstance(flags, list) or not all(isinstance(flag, str) for flag in flags):
            raise ValueError(
                f"{MODELS_CONFIG_PATH}: device_profiles.{device}.flags must be a string list"
            )
        device_profiles[str(device)] = tuple(flags)
    for device in ("cpu", "gpu"):
        if device not in device_profiles:
            raise ValueError(f"{MODELS_CONFIG_PATH}: device_profiles.{device} is not defined")

    default_device = str(raw.get("default_device") or "auto")
    if default_device not in ("auto", "cpu", "gpu"):
        raise ValueError(f"{MODELS_CONFIG_PATH}: default_device must be auto, cpu or gpu")

    return Catalog(
        root=root,
        talkers=talkers,
        defaults=defaults,
        codecs=codecs,
        device_profiles=device_profiles,
        default_device=default_device,
    )


def _print_catalog(catalog: Catalog) -> None:
    print(f"Catalog : {MODELS_CONFIG_PATH}")
    print(f"Models  : {catalog.root}")
    for role, title in ((SERVER_ROLE, "Serve"), (DESIGN_ROLE, "Design")):
        print(f"\n{title} candidates:")
        models = catalog.for_role(role, available_only=False)
        if not models:
            print("  (none configured)")
        for model in models:
            state = "ready" if not model.missing else (
                "MISSING " + ", ".join(path.name for path in model.missing)
            )
            default = " (default)" if catalog.defaults.get(role) == model.id else ""
            print(
                f"  {model.id:<24} {model.label:<30} {model.quant:<7} "
                f"{model.model_type:<12} {state}{default}"
            )
            if model.note:
                print(f"  {'':<24} {model.note}")
    print("\nCodecs:")
    for quant, path in catalog.codecs.items():
        state = "" if path.is_file() else "  MISSING"
        print(f"  {quant:<7} {path.name}{state}")
    print("\nDevice profiles:")
    for device, flags in catalog.device_profiles.items():
        print(f"  {device:<5} {' '.join(flags)}")
    print(f"  default_device: {catalog.default_device}")


def _setup_logging(level_name: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"sylvia-server-{datetime.now():%Y%m%d}.log"
    level = getattr(logging, level_name.upper(), logging.INFO)
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.propagate = False
    for handler in list(LOGGER.handlers):
        LOGGER.removeHandler(handler)
        handler.close()
    stamp = "%(asctime)s %(levelname)-7s %(message)s"
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(stamp))
    LOGGER.addHandler(file_handler)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(stamp, "%H:%M:%S"))
    LOGGER.addHandler(console)
    LOGGER.debug("logging to %s (level %s)", log_path, level_name.upper())
    return log_path


@dataclass(frozen=True)
class Backend:
    """A runnable server type: which venv supplies the native runtime."""

    device: str
    label: str
    venv: str
    dist: str
    note: str


BACKENDS = (
    Backend(
        device="cpu",
        label="CPU server",
        venv=".venv",
        dist="realtimetts-qwen-native-cpu",
        note="6 generation + 6 codec threads; works without a GPU",
    ),
    Backend(
        device="gpu",
        label="GPU server (CUDA)",
        venv=".venv-gpu",
        dist="realtimetts-qwen-native",
        note="NVIDIA GPU; low latency, best with 0.6B Q8 or 1.7B",
    ),
)


def _backend_python(backend: Backend) -> Path:
    return ROOT / backend.venv / "Scripts" / "python.exe"


def _probe_backend(backend: Backend) -> tuple[bool, str]:
    python = _backend_python(backend)
    if not python.is_file():
        return False, f"{backend.venv} not installed"
    probe = (
        "import importlib.metadata as m\n"
        f"print(m.version({backend.dist!r}))\n"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", probe], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"probe failed: {exc}"
    if result.returncode != 0:
        lines = (result.stderr or "").strip().splitlines()
        return False, lines[-1] if lines else f"exit {result.returncode}"
    version = result.stdout.strip()
    if backend.device == "gpu" and not _gpu_present():
        return False, f"{backend.dist} {version}, but no NVIDIA GPU visible"
    return True, f"{backend.dist} {version}"


def _print_backends() -> None:
    print("\nServer types:")
    for backend in BACKENDS:
        ready, detail = _probe_backend(backend)
        state = "ready" if ready else "UNAVAILABLE"
        print(f"  {backend.device:<4} {backend.label:<20} {backend.venv:<10} {state}  ({detail})")


def _log_gpu_memory() -> None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
             "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        LOGGER.warning("gpu memory probe failed: %s", exc)
        return
    if result.returncode != 0:
        LOGGER.warning("gpu memory probe failed: %s", result.stderr.strip())
        return
    for line in result.stdout.strip().splitlines():
        LOGGER.info("gpu memory (used, free, total): %s", line.strip())


def _choose_backend(requested: str, assume_default: bool) -> tuple[Backend, bool, str]:
    """Return (backend, ready, detail). Prompts when more than one backend works."""
    probes = [(backend, *_probe_backend(backend)) for backend in BACKENDS]
    ready = [item for item in probes if item[1]]

    if requested != "auto":
        chosen = next(item for item in probes if item[0].device == requested)
        if not chosen[1]:
            raise RuntimeError(
                f"server type {requested!r} is unavailable: {chosen[2]}. "
                f"Install it with {' '.join(_install_hint(chosen[0]))}"
            )
        LOGGER.info("server type %s selected explicitly", requested)
        return chosen[0], True, chosen[2]

    if not ready:
        raise RuntimeError(
            "no server type is installed; run Setup-TTS.bat (.venv) or Setup-TTS-GPU.bat (.venv-gpu)"
        )

    print("\nChoose the server type:")
    default = next((item[0] for item in ready if item[0].device == "gpu"), ready[0][0])
    for index, (backend, is_ready, detail) in enumerate(probes, 1):
        state = "ready" if is_ready else f"unavailable ({detail})"
        marker = " *" if is_ready and backend is default else ""
        print(f"  [{index}] {backend.label:<20} {backend.venv:<10} {state}{marker}")
        print(f"      {backend.note}")

    if assume_default:
        LOGGER.info("server type default: %s", default.device)
        return default, True, next(detail for backend, ok, detail in probes if backend is default)
    if len(probes) == 1:
        return default, True, next(detail for backend, ok, detail in probes if backend is default)

    detail_of_default = next(detail for backend, ok, detail in probes if backend is default)
    while True:
        try:
            answer = input(f"server type [Enter = {default.device}]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return default, True, detail_of_default
        if not answer:
            return default, True, detail_of_default
        if answer in {"q", "quit", "exit"}:
            raise SystemExit("Cancelled.")
        for index, (backend, is_ready, detail) in enumerate(probes, 1):
            if answer == backend.device or (answer.isdigit() and int(answer) == index):
                if not is_ready:
                    print(f"  {backend.label} is unavailable: {detail}")
                    break
                return backend, True, detail
        else:
            print(f"  {answer!r} is not a choice; enter cpu, gpu, a number, or Enter.")


def _install_hint(backend: Backend) -> tuple[str, ...]:
    if backend.device == "gpu":
        return ("Setup-TTS-GPU.bat",)
    return ("Setup-TTS.bat",)


def _reexec(backend: Backend, resolved: list[str], args: argparse.Namespace) -> int:
    """Hand off to the interpreter that owns the chosen native runtime."""
    python = _backend_python(backend)
    argv = [str(python), str(Path(__file__).resolve()), "--device", backend.device, *resolved]
    if args.new_reference:
        argv.append("-new")
    if args.dry_run:
        argv.append("--dry-run")
    if args.assume_yes:
        argv.append("--yes")
    argv.extend(("--log-level", args.log_level))
    LOGGER.info("handing off to %s server: %s", backend.device, python)
    LOGGER.debug("handoff argv: %s", argv)
    return subprocess.call(argv, cwd=str(ROOT))


def _gpu_present() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        LOGGER.warning("nvidia-smi unavailable: %s", exc)
        return False
    if result.returncode != 0:
        LOGGER.warning("nvidia-smi exited %s: %s", result.returncode, result.stderr.strip())
        return False
    report = result.stdout.strip()
    if report:
        LOGGER.info("GPU detected: %s", report.replace("\n", "; "))
    return bool(report)


def _read_choice(question: str, options: list[Talker], default: Talker) -> Talker:
    while True:
        try:
            answer = input(question).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return default
        if not answer:
            return default
        if answer.lower() in {"q", "quit", "exit"}:
            raise SystemExit("Cancelled.")
        match = next((model for model in options if model.id.lower() == answer.lower()), None)
        if match is not None:
            return match
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        print(
            f"  {answer!r} is not a choice; enter a number, an id, "
            f"or press Enter for {default.id}."
        )


def _choose_model(
    catalog: Catalog,
    role: str,
    requested_id: str | None,
    *,
    assume_default: bool,
) -> Talker:
    if requested_id:
        model = catalog.talkers.get(requested_id)
        if model is None:
            known = ", ".join(catalog.talkers)
            raise ValueError(f"Unknown {role} model id {requested_id!r}; catalog has: {known}")
        if role not in model.roles:
            raise ValueError(f"Model {model.id!r} is not usable for {role!r}")
        if model.missing:
            raise FileNotFoundError(
                "Model files are missing:\n" + "\n".join(str(path) for path in model.missing)
            )
        return model

    options = catalog.for_role(role)
    if not options:
        raise FileNotFoundError(
            f"No {role} model is available under {catalog.root}; check .config/models.json"
        )
    default = catalog.default_for(role) or options[0]
    if assume_default:
        print(f"Using {role} model: {default.id} ({default.quant})")
        return default

    print(f"\nChoose the {role} model ({len(options)} available):")
    for index, model in enumerate(options, 1):
        flags = " (default)" if model.id == default.id else ""
        print(
            f"  [{index}] {model.label:<30} {model.quant:<7} "
            f"{model.size_mb:8.0f} MB  {model.model_type}{flags}"
        )
        if model.note:
            print(f"      {model.note}")
    chosen = _read_choice(f"{role} model [Enter = {default.id}]: ", options, default)
    verb = "Serving" if role == SERVER_ROLE else "Using"
    print(f"{verb} {chosen.id} ({chosen.quant})")
    return chosen


def _require_asset(model: Talker) -> None:
    missing = model.missing
    if missing:
        raise FileNotFoundError("Required GGUF files are missing:\n" + "\n".join(map(str, missing)))


def _seed_script() -> str:
    if not VOICE_PROFILE_PATH.is_file():
        raise FileNotFoundError(VOICE_PROFILE_PATH)
    script = " ".join(SEED_SCRIPT_PATH.read_text(encoding="utf-8").split())
    if not script:
        raise ValueError(f"{SEED_SCRIPT_PATH} must not be empty")
    return script


def _write_reference_files(audio: bytes, script: str) -> tuple[float, int]:
    import soundfile as sf

    script = " ".join(script.split())
    if not audio:
        raise ValueError("Generated Sylvia reference is empty")
    if not script:
        raise ValueError("Sylvia reference transcript is empty")

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    temporary_audio = REFERENCE_PATH.with_suffix(".wav.tmp")
    temporary_text = REFERENCE_TEXT_PATH.with_suffix(".txt.tmp")
    try:
        temporary_audio.write_bytes(audio)
        info = sf.info(str(temporary_audio))
        if info.format != "WAV" or info.frames <= 0:
            raise ValueError("Sylvia reference is not a valid, non-empty WAV")
        temporary_text.write_text(script, encoding="utf-8")
        temporary_text.replace(REFERENCE_TEXT_PATH)
        temporary_audio.replace(REFERENCE_PATH)
        return info.duration, info.samplerate
    finally:
        temporary_audio.unlink(missing_ok=True)
        temporary_text.unlink(missing_ok=True)


def _prepare_reference_folders() -> None:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    VOICE_DIR.mkdir(parents=True, exist_ok=True)

    if REFERENCE_PATH.exists() and not REFERENCE_PATH.is_file():
        raise ValueError(f"Reference path is not a file: {REFERENCE_PATH}")
    if REFERENCE_PATH.is_file() and REFERENCE_PATH.stat().st_size > 0:
        return

    from RealtimeTTS.qwen_server import VoiceRegistry

    record = VoiceRegistry(VOICE_DIR).get(VOICE_NAME)
    if record is None:
        return
    if record.kind != "wav" or len(record.files) != 1 or not record.ref_text.strip():
        raise ValueError(f"Existing {VOICE_NAME!r} registry entry cannot seed a WAV reference")

    saved_audio = VOICE_DIR / record.files[0]
    if not saved_audio.is_file():
        raise FileNotFoundError(saved_audio)
    _write_reference_files(saved_audio.read_bytes(), record.ref_text)
    print(f"Restored existing Sylvia reference to: {REFERENCE_PATH}")


def _load_reference() -> tuple[bytes, str] | None:
    if not REFERENCE_PATH.is_file() or REFERENCE_PATH.stat().st_size == 0:
        return None
    if not REFERENCE_TEXT_PATH.is_file():
        raise ValueError(
            f"Reference transcript is missing: {REFERENCE_TEXT_PATH}; use -new to regenerate"
        )

    import soundfile as sf

    info = sf.info(str(REFERENCE_PATH))
    if info.format != "WAV" or info.frames <= 0:
        raise ValueError(f"Reference is not a valid, non-empty WAV: {REFERENCE_PATH}")
    script = " ".join(REFERENCE_TEXT_PATH.read_text(encoding="utf-8").split())
    if not script:
        raise ValueError(f"Reference transcript is empty: {REFERENCE_TEXT_PATH}")
    return REFERENCE_PATH.read_bytes(), script


def _archive_reference() -> None:
    paths = (REFERENCE_PATH, REFERENCE_TEXT_PATH)
    existing = [path for path in paths if path.is_file()]
    if not existing:
        return

    ARCHIVE_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    for path in existing:
        archived_path = ARCHIVE_REFERENCE_DIR / f"sylvia-{timestamp}{path.suffix}"
        path.replace(archived_path)
        print(f"Archived reference: {archived_path}")


def _assert_port_free(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Port {port} is already in use; stop its server before starting this pipeline") from exc


def _server_command(
    model: Talker,
    device_flags: tuple[str, ...],
    host: str,
    port: int,
    *,
    interpreter: str | None = None,
    warmup_voice: str | None = None,
    clone_mode: str | None = None,
) -> list[str]:
    command = [
        interpreter or sys.executable,
        str(ROOT / "run_qwen_server.py"),
        *COMMON_FLAGS,
        *device_flags,
        "--alias", model.alias,
        "--quant", model.quant,
        "--model", str(model.path),
        "--codec", str(model.codec_path),
        "--host", host,
        "--port", str(port),
        "--voice-dir", str(VOICE_DIR),
    ]
    if warmup_voice:
        command.extend(("--startup-warmup-voice", warmup_voice))
    if clone_mode:
        command.extend(("--clone-mode", clone_mode))
    if host != "127.0.0.1":
        command.append("--allow-lan")
    return command


def _wait_until_ready(process: subprocess.Popen[Any], port: int) -> None:
    deadline = time.monotonic() + 1800
    url = f"http://127.0.0.1:{port}/ready"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"VoiceDesign server exited during startup (code {process.returncode})")
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                status = json.loads(response.read().decode("utf-8")).get("status")
                if response.status == 200 and status == "ready":
                    return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(2)
    raise TimeoutError("VoiceDesign server did not become ready within 30 minutes")


def _generate_reference(script: str, model: Talker, device_flags: tuple[str, ...], device: str) -> bytes:
    _assert_port_free(DESIGN_PORT)
    command = _server_command(
        model,
        device_flags,
        "127.0.0.1",
        DESIGN_PORT,
        warmup_voice="design",
    )
    LOGGER.info("design stage: %s %s on %s", model.id, model.quant, device)
    LOGGER.debug("design command: %s", command)
    print(f"Loading {model.label} {model.quant} on {device.upper()} to build Sylvia's reference audio...")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    try:
        _wait_until_ready(process, DESIGN_PORT)
        LOGGER.info("design stage ready; synthesizing %s characters", len(script))
        print(f"Generating reference from the {len(script)}-character seed script...")
        request = urllib.request.Request(
            f"http://127.0.0.1:{DESIGN_PORT}/v1/audio/speech",
            data=json.dumps({
                "model": model.alias,
                "voice": "design",
                "language": "English",
                "input": script,
                "response_format": "wav",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                if response.status != 200:
                    raise RuntimeError(f"VoiceDesign returned HTTP {response.status}")
                audio = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            LOGGER.error("design request failed HTTP %s: %s", exc.code, detail[:500])
            raise RuntimeError(f"VoiceDesign request failed (HTTP {exc.code}): {detail}") from exc
        if not audio:
            raise RuntimeError("VoiceDesign returned an empty WAV")
        LOGGER.info(
            "design stage produced %s bytes in %.1fs",
            len(audio),
            time.monotonic() - started,
        )
        return audio
    finally:
        _stop_stage_server(process)


def _stop_stage_server(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.CTRL_BREAK_EVENT)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        command = [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            f"Stop-Process -Id {process.pid} -Force",
        ]
        subprocess.run(command, check=True)
        process.wait(timeout=15)
    print("VoiceDesign stage stopped; releasing its model memory.")


def _register_reference(audio: bytes, script: str) -> None:
    from RealtimeTTS.qwen_server import VoiceRegistry

    registry = VoiceRegistry(VOICE_DIR)
    current = registry.get(VOICE_NAME)
    audio_hash = hashlib.sha256(audio).hexdigest()
    if (
        current is not None
        and current.kind == "wav"
        and current.ref_text == script
        and current.sha256 == (audio_hash,)
    ):
        return
    record = registry.candidate(VOICE_NAME, script, "wav", (audio,))
    registry.commit(record)


def _save_and_register_reference(audio: bytes, script: str) -> None:
    duration, samplerate = _write_reference_files(audio, script)
    _register_reference(audio, script)
    print(f"Saved Sylvia reference: {REFERENCE_PATH} ({duration:.1f}s, {samplerate} Hz)")


def _print_command(title: str, command: list[str]) -> None:
    print(f"\n{title}:")
    print("  " + " ".join(f'"{part}"' if " " in part else part for part in command))


def main() -> int:
    args = _parse_args()
    if os.name != "nt":
        raise OSError("This two-stage launcher uses Windows CPU core placement and must run on Windows")

    log_path = _setup_logging(args.log_level)
    LOGGER.info("=== Sylvia TTS launcher (pid %s, %s) ===", os.getpid(), sys.executable)
    LOGGER.info("log file: %s", log_path)

    catalog = _load_catalog()
    LOGGER.info("catalog: %s | models: %s", MODELS_CONFIG_PATH, catalog.root)
    if args.list_models:
        _print_catalog(catalog)
        _print_backends()
        return 0

    backend, _, backend_detail = _choose_backend(args.device, args.assume_yes)
    LOGGER.info("server type: %s (%s)", backend.device, backend_detail)
    device_flags = catalog.device_profiles[backend.device]
    LOGGER.debug("device flags: %s", " ".join(device_flags))
    if backend.device == "gpu":
        _log_gpu_memory()

    serve_model = _choose_model(
        catalog, SERVER_ROLE, args.serve_model, assume_default=args.assume_yes
    )
    LOGGER.info("serve model: %s (%s, codec %s)", serve_model.id, serve_model.quant, serve_model.codec_quant)
    _require_asset(serve_model)

    needs_reference = args.new_reference or not (
        REFERENCE_PATH.is_file() and REFERENCE_PATH.stat().st_size > 0
    )
    design_model: Talker | None = None
    if needs_reference:
        design_model = _choose_model(
            catalog, DESIGN_ROLE, args.design_model, assume_default=args.assume_yes
        )
        _require_asset(design_model)
        LOGGER.info("design model: %s (%s)", design_model.id, design_model.quant)
        if backend.device == "cpu" and design_model.quant in {"BF16", "Q8_0"}:
            LOGGER.warning(
                "%s design model on CPU is very slow; the gpu server type is much faster",
                design_model.quant,
            )

    if _backend_python(backend) != Path(sys.executable):
        resolved = ["--model", serve_model.id]
        if design_model is not None:
            resolved.extend(("--design-model", design_model.id))
        LOGGER.info("this interpreter is not %s; handing off", backend.venv)
        return _reexec(backend, resolved, args)

    serve_command = _server_command(
        serve_model,
        device_flags,
        "0.0.0.0",
        SERVER_PORT,
        warmup_voice=VOICE_NAME,
        clone_mode="auto",
    )

    if args.dry_run:
        _print_command("Serve command", serve_command)
        if design_model is not None:
            _print_command(
                "Design command",
                _server_command(design_model, device_flags, "127.0.0.1", DESIGN_PORT, warmup_voice="design"),
            )
        print(f"\nDry run: no server started. Device: {backend.device} ({backend_detail})")
        return 0

    _prepare_reference_folders()

    if needs_reference:
        if design_model is None:
            raise RuntimeError("a design model is required to build the reference")
        script = _seed_script()
        _assert_port_free(SERVER_PORT)
        _assert_port_free(DESIGN_PORT)
        if args.new_reference:
            _archive_reference()
        print("\nPipeline: design the reference, then serve it.")
        _print_command("Serve command", serve_command)
        started = time.monotonic()
        reference = _generate_reference(script, design_model, device_flags, backend.device)
        _save_and_register_reference(reference, script)
        LOGGER.info("reference pipeline finished in %.1fs", time.monotonic() - started)
    else:
        saved_reference = _load_reference()
        if saved_reference is None:
            raise RuntimeError(f"reference {REFERENCE_PATH} is missing or empty")
        reference, script = saved_reference
        _register_reference(reference, script)
        LOGGER.info("reusing saved reference: %s", REFERENCE_PATH)
        print(f"Reusing saved Sylvia reference: {REFERENCE_PATH}; skipping VoiceDesign.")
        _assert_port_free(SERVER_PORT)
        _print_command("Serve command", serve_command)

    LOGGER.info(
        "starting %s %s (codec %s) on %s host 0.0.0.0:%s",
        serve_model.id,
        serve_model.quant,
        serve_model.codec_quant,
        backend.device,
        SERVER_PORT,
    )
    print(
        f"\nStarting {serve_model.label} {serve_model.quant} (codec {serve_model.codec_quant}) "
        f"on {backend.device.upper()} with the Sylvia voice..."
    )
    exit_code = subprocess.call(serve_command, cwd=ROOT)
    LOGGER.info("server exited with code %s", exit_code)
    if exit_code != 0:
        LOGGER.error("check the log above for the server's own diagnostics: %s", log_path)
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        logging.getLogger("sylvia.server").error("startup failed: %s", exc)
        print(f"Startup failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
