"""
Kokoro synthesis helpers: one sentence in, one trimmed buffer out.

Kokoro pads every call with roughly 300 ms of leading and 500 ms of trailing
near-silence. `trim_padding` cuts that to KEEP_MS so per-sentence audio joins
without dead air but keeps a natural pause.
"""

from __future__ import annotations

import numpy as np

TTS_SAMPLE_RATE = 24_000

# Near-silence floor for edge detection. Kokoro speech sits well above this.
SILENCE_RMS = 0.005
SILENCE_WINDOW = 240  # 10 ms at 24 kHz

# Calibrated: at 200 ms the per-sentence rendering of a six sentence passage
# totals 25.83 s against 25.73 s for the same text synthesised in one call.
KEEP_MS = 200.0


def edge_silence(audio: np.ndarray) -> tuple[float, float]:
    """Leading and trailing near-silence in ms, measured in 10 ms windows."""
    if len(audio) == 0:
        return 0.0, 0.0

    windows = len(audio) // SILENCE_WINDOW
    if windows == 0:
        return 0.0, 0.0

    frames = audio[: windows * SILENCE_WINDOW].reshape(windows, SILENCE_WINDOW)
    loud = np.sqrt(np.mean(np.square(frames), axis=1)) >= SILENCE_RMS

    if not loud.any():
        return round(len(audio) / TTS_SAMPLE_RATE * 1000, 1), 0.0

    first = int(np.argmax(loud))
    last = int(len(loud) - 1 - np.argmax(loud[::-1]))
    per_window_ms = SILENCE_WINDOW / TTS_SAMPLE_RATE * 1000

    return (
        round(first * per_window_ms, 1),
        round((len(loud) - 1 - last) * per_window_ms, 1),
    )


def trim_padding(audio: np.ndarray, keep_ms: float = KEEP_MS) -> np.ndarray:
    """Drop Kokoro's per-call edge padding, keeping `keep_ms` of breath."""
    lead_ms, tail_ms = edge_silence(audio)

    start = max(0, int((lead_ms - keep_ms) / 1000 * TTS_SAMPLE_RATE))
    end = len(audio) - max(0, int((tail_ms - keep_ms) / 1000 * TTS_SAMPLE_RATE))

    return audio[start:end] if end > start else audio


def synthesize(kokoro, text: str, voice: str) -> np.ndarray:
    """One sentence in, one buffer out."""
    chunks: list[np.ndarray] = []

    for result in kokoro(text, voice=voice, speed=1.0):
        audio = result.audio if hasattr(result, "audio") else result[2]
        if audio is None:
            continue
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        elif hasattr(audio, "numpy"):
            audio = audio.numpy()
        chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))

    if not chunks:
        raise RuntimeError(f"Kokoro produced no audio for: {text!r}")

    return np.concatenate(chunks)
