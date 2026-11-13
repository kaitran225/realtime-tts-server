"""Generate a Sylvia reference with VoiceDesign, then serve it from Base Q8."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / ".config"
MODEL_DIR = Path(r"D:\Shared\GGUF\Serveurperso\Qwen3-TTS-GGUF")
VOICE_DIR = ROOT / "voices"
REFERENCE_DIR = ROOT / "reference"
ARCHIVE_REFERENCE_DIR = ROOT / "archive-reference"
REFERENCE_PATH = REFERENCE_DIR / "sylvia_reference.wav"
REFERENCE_TEXT_PATH = REFERENCE_DIR / "sylvia_reference.txt"
SEED_SCRIPT_PATH = CONFIG_DIR / "sylvia-seed-script.txt"
DESIGN_PORT = 2559
SERVER_PORT = 2558
DESIGN_MODEL_Q4 = MODEL_DIR / "qwen-talker-1.7b-voicedesign-Q4_K_M.gguf"
DESIGN_CODEC_Q4 = MODEL_DIR / "qwen-tokenizer-12hz-Q4_K_M.gguf"
DESIGN_MODEL_BF16 = MODEL_DIR / "qwen-talker-1.7b-voicedesign-BF16.gguf"
DESIGN_CODEC_BF16 = MODEL_DIR / "qwen-tokenizer-12hz-BF16.gguf"
BASE_MODEL_Q4 = MODEL_DIR / "qwen-talker-0.6b-base-Q4_K_M.gguf"
BASE_CODEC_Q4 = MODEL_DIR / "qwen-tokenizer-12hz-Q4_K_M.gguf"
BASE_MODEL_Q8 = MODEL_DIR / "qwen-talker-0.6b-base-Q8_0.gguf"
BASE_CODEC_Q8 = MODEL_DIR / "qwen-tokenizer-12hz-Q8_0.gguf"
VOICE_NAME = "sylvia"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and serve the Sylvia voice.")
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="use BF16 VoiceDesign assets instead of the default Q4_K_M assets",
    )
    parser.add_argument(
        "-new",
        "--new",
        dest="new_reference",
        action="store_true",
        help="archive the current reference and generate a new one",
    )
    parser.add_argument(
        "--Q8",
        dest="base_q8",
        action="store_true",
        help="serve with the 0.6B Base Q8 model (default: Q4_K_M)",
    )
    return parser.parse_args(argv)


def _design_assets(use_bf16: bool) -> tuple[Path, Path, str]:
    if use_bf16:
        return DESIGN_MODEL_BF16, DESIGN_CODEC_BF16, "BF16"
    return DESIGN_MODEL_Q4, DESIGN_CODEC_Q4, "Q4_K_M"


def _require_files(
    design_assets: tuple[Path, Path, str] | None,
    base_model: Path,
    base_codec: Path,
) -> str | None:
    missing = [path for path in (base_model, base_codec) if not path.is_file()]
    if design_assets:
        design_model, design_codec, _ = design_assets
        missing.extend(path for path in (design_model, design_codec) if not path.is_file())
    if missing:
        raise FileNotFoundError("Required GGUF files are missing:\n" + "\n".join(map(str, missing)))
    if not (CONFIG_DIR / "voice-profile.txt").is_file():
        raise FileNotFoundError(CONFIG_DIR / "voice-profile.txt")
    if not design_assets:
        return None
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
    talker: Path,
    codec: Path,
    quant: str,
    alias: str,
    host: str,
    port: int,
    *,
    warmup_voice: str | None = None,
    clone_mode: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "run_qwen_server.py"),
        "--max-active-requests", "1",
        "--max-queued-requests", "0",
        "--device", "cpu",
        "--cpu-threads", "6",
        "--cpu-codec-threads", "6",
        "--cpu-stream-frames", "1",
        "--cpu-fused-attention",
        "--startup-buffer-ms", "80",
        "--synthesis-timeout", "1800",
        "--alias", alias,
        "--quant", quant,
        "--lang", "English",
        "--model", str(talker),
        "--codec", str(codec),
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


def _generate_reference(
    script: str,
    design_model: Path,
    design_codec: Path,
    design_quant: str,
) -> bytes:
    _assert_port_free(DESIGN_PORT)
    command = _server_command(
        design_model,
        design_codec,
        design_quant,
        "sylvia-voice-design-stage",
        "127.0.0.1",
        DESIGN_PORT,
        warmup_voice="design",
    )
    print(f"Loading 1.7B VoiceDesign {design_quant} on CPU to build Sylvia's reference audio...")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    try:
        _wait_until_ready(process, DESIGN_PORT)
        print(f"Generating reference from the {len(script)}-character seed script...")
        request = urllib.request.Request(
            f"http://127.0.0.1:{DESIGN_PORT}/v1/audio/speech",
            data=json.dumps({
                "model": "sylvia-voice-design-stage",
                "voice": "design",
                "language": "English",
                "input": script,
                "response_format": "wav",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                if response.status != 200:
                    raise RuntimeError(f"VoiceDesign returned HTTP {response.status}")
                audio = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"VoiceDesign request failed (HTTP {exc.code}): {detail}") from exc
        if not audio:
            raise RuntimeError("VoiceDesign returned an empty WAV")
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


def main() -> int:
    args = _parse_args()
    if os.name != "nt":
        raise OSError("This two-stage launcher uses Windows CPU core placement and must run on Windows")
    _prepare_reference_folders()
    if args.new_reference:
        saved_reference = None
    else:
        saved_reference = _load_reference()

    design_assets = _design_assets(args.bf16) if saved_reference is None else None
    base_model, base_codec, base_quant = (
        (BASE_MODEL_Q8, BASE_CODEC_Q8, "Q8_0")
        if args.base_q8
        else (BASE_MODEL_Q4, BASE_CODEC_Q4, "Q4_K_M")
    )
    base_alias = (
        "qwen3-tts-0.6b-base-q8-sylvia"
        if args.base_q8
        else "qwen3-tts-0.6b-base-q4-sylvia"
    )
    script = _require_files(design_assets, base_model, base_codec)
    _assert_port_free(SERVER_PORT)
    if saved_reference is None:
        if design_assets is None or script is None:
            raise RuntimeError("VoiceDesign assets or seed script are unavailable")
        _assert_port_free(DESIGN_PORT)
        if args.new_reference:
            _archive_reference()
        design_model, design_codec, design_quant = design_assets
        reference = _generate_reference(script, design_model, design_codec, design_quant)
        _save_and_register_reference(reference, script)
    else:
        reference, script = saved_reference
        _register_reference(reference, script)
        print(f"Reusing saved Sylvia reference: {REFERENCE_PATH}; skipping VoiceDesign.")

    command = _server_command(
        base_model,
        base_codec,
        base_quant,
        base_alias,
        "0.0.0.0",
        SERVER_PORT,
        warmup_voice=VOICE_NAME,
        clone_mode="auto",
    )
    print(f"Starting 0.6B Base {base_quant} server with the saved Sylvia voice...")
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
