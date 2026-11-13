"""Run the unauthenticated LAN server with the default VoiceDesign profile."""

import json
from pathlib import Path

from RealtimeTTS import qwen_server

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / ".config"
PROFILE_PATH = CONFIG_PATH / "voice-profile.txt"
VOICE_PROFILE = PROFILE_PATH.read_text(encoding="utf-8").strip()
if not VOICE_PROFILE:
    raise ValueError(f"Voice profile is empty: {PROFILE_PATH}")


def _validate_bind_without_api_key(args, _api_key, parser):
    if not qwen_server._is_loopback_host(args.host) and not args.allow_lan:
        parser.error("--allow-lan is required when --host is not loopback")


def _install_voice_profile():
    original = qwen_server.QwenHttpServer.parse_speech_options

    def parse_speech_options(server, payload):
        if getattr(server.engine, "model_type", None) != "voice_design":
            return original(server, payload)
        configured = dict(payload)
        if not configured.get("voice"):
            configured["voice"] = "design"
        requested_instructions = str(configured.get("instructions") or "").strip()
        configured["instructions"] = VOICE_PROFILE
        if requested_instructions and requested_instructions != VOICE_PROFILE:
            configured["instructions"] += (
                "\n\nAdditional speech delivery direction: " + requested_instructions
            )
        return original(server, configured)

    qwen_server.QwenHttpServer.parse_speech_options = parse_speech_options


def _install_studio_default_prompt():
    studio_js = Path(qwen_server.__file__).with_name("studio") / "studio.js"
    source = studio_js.read_text(encoding="utf-8")
    profile_assignment = (
        "window.SYLVIA_VOICE_PROFILE = "
        + json.dumps(VOICE_PROFILE, ensure_ascii=True)
        + ";"
    )
    lines = source.splitlines()
    assignment_index = next(
        (i for i, line in enumerate(lines) if line.startswith("window.SYLVIA_VOICE_PROFILE = ")),
        None,
    )
    if assignment_index is not None:
        lines[assignment_index] = profile_assignment
    else:
        numeric_index = next(
            (i for i, line in enumerate(lines) if line.startswith("const numeric = ")),
            None,
        )
        if numeric_index is None:
            raise RuntimeError(f"Unsupported RealtimeTTS Studio asset: {studio_js}")
        lines.insert(numeric_index, profile_assignment)
    source = "\n".join(lines) + "\n"

    capabilities_load = "caps=await(await api('/v1/capabilities')).json();"
    if capabilities_load not in source:
        raise RuntimeError(f"Unsupported RealtimeTTS Studio connect flow: {studio_js}")
    prefill = (
        "if(caps.features?.model_type==='voice_design'&&"
        "!$('instructions').value.trim())"
        "$('instructions').value=window.SYLVIA_VOICE_PROFILE;"
    )
    if prefill not in source:
        source = source.replace(capabilities_load, capabilities_load + prefill, 1)
    studio_js.write_text(source, encoding="utf-8")


def main():
    if not hasattr(qwen_server, "_validate_bind_security"):
        raise RuntimeError("Unsupported RealtimeTTS version: LAN bind validation was not found")
    qwen_server._validate_bind_security = _validate_bind_without_api_key
    _install_voice_profile()
    _install_studio_default_prompt()
    qwen_server.main()


if __name__ == "__main__":
    main()
