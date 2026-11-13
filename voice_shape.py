"""Timbre shaping for the generated voice reference (pure stdlib).

The VoiceDesign model picks its own brightness; a gentle shelf pair pulls the
reference toward a darker, chestier target before it is used for cloning.

Config lives in models.json -> "reference_shape":
    {"low_shelf": {"freq": 200, "gain_db": 2.0},
     "peak": {"freq": 900, "gain_db": 0.0, "q": 0.8},
     "high_shelf": {"freq": 3200, "gain_db": -6.0}}
"""

from __future__ import annotations

import array
import io
import math
import sys
import wave


def _biquad(kind: str, rate: float, freq: float, q: float = 0.707, gain_db: float = 0.0):
    freq = max(10.0, min(freq, rate * 0.45))
    w0 = 2.0 * math.pi * freq / rate
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / (2.0 * max(q, 0.05))
    amp = 10.0 ** (gain_db / 40.0)

    if kind == "peak":
        b0 = 1.0 + alpha * amp
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * amp
        a0 = 1.0 + alpha / amp
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / amp
    elif kind == "low_shelf":
        two = 2.0 * math.sqrt(amp) * alpha
        b0 = amp * ((amp + 1) - (amp - 1) * cos_w0 + two)
        b1 = 2 * amp * ((amp - 1) - (amp + 1) * cos_w0)
        b2 = amp * ((amp + 1) - (amp - 1) * cos_w0 - two)
        a0 = (amp + 1) + (amp - 1) * cos_w0 + two
        a1 = -2 * ((amp - 1) + (amp + 1) * cos_w0)
        a2 = (amp + 1) + (amp - 1) * cos_w0 - two
    elif kind == "high_shelf":
        two = 2.0 * math.sqrt(amp) * alpha
        b0 = amp * ((amp + 1) + (amp - 1) * cos_w0 + two)
        b1 = -2 * amp * ((amp - 1) + (amp + 1) * cos_w0)
        b2 = amp * ((amp + 1) + (amp - 1) * cos_w0 - two)
        a0 = (amp + 1) - (amp - 1) * cos_w0 + two
        a1 = 2 * ((amp - 1) - (amp + 1) * cos_w0)
        a2 = (amp + 1) - (amp - 1) * cos_w0 - two
    else:
        raise ValueError(f"unknown biquad {kind!r}")
    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


def _apply(samples: list[float], coeffs) -> None:
    b0, b1, b2, a1, a2 = coeffs
    x1 = x2 = y1 = y2 = 0.0
    for index, sample in enumerate(samples):
        value = b0 * sample + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1 = x1, sample
        y2, y1 = y1, value
        samples[index] = value


def shape_wav(data: bytes, shape: dict) -> tuple[bytes, dict]:
    """Apply shelf/peak shaping to a PCM16 mono WAV. Returns (wav, stats)."""
    if not shape:
        return data, {"skipped": True}

    with wave.open(io.BytesIO(data), "rb") as reader:
        channels = reader.getnchannels()
        width = reader.getsampwidth()
        rate = reader.getframerate()
        frames = reader.readframes(reader.getnframes())
    if width != 2:
        return data, {"skipped": True, "reason": f"{width * 8}-bit"}

    raw = array.array("h")
    raw.frombytes(frames)
    if sys.byteorder == "big":
        raw.byteswap()

    per_channel: list[list[float]] = [[] for _ in range(channels)]
    for index, sample in enumerate(raw):
        per_channel[index % channels].append(sample / 32768.0)

    applied = []
    for kind in ("low_shelf", "peak", "high_shelf"):
        band = shape.get(kind) or {}
        gain = float(band.get("gain_db") or 0.0)
        if abs(gain) < 0.01:
            continue
        coeffs = _biquad(
            kind, rate,
            float(band.get("freq") or 1000.0),
            float(band.get("q") or 0.707),
            gain,
        )
        for channel in per_channel:
            _apply(channel, coeffs)
        applied.append({"kind": kind, "freq": band.get("freq"), "gain_db": gain})

    peak = max((abs(value) for channel in per_channel for value in channel), default=0.0)
    scale = 0.999 / peak if peak > 0.999 else 1.0

    output = array.array("h")
    for frame_index in range(len(per_channel[0])):
        for channel in per_channel:
            value = channel[frame_index] * scale
            value = 1.0 if value > 1.0 else (-1.0 if value < -1.0 else value)
            output.append(int(value * 32767.0))
    if sys.byteorder == "big":
        output.byteswap()

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(output.tobytes())

    return buffer.getvalue(), {
        "bands": applied,
        "peak_before": round(peak, 3),
        "limiter_scale": round(scale, 3),
        "seconds": round(len(per_channel[0]) / rate, 2),
    }
