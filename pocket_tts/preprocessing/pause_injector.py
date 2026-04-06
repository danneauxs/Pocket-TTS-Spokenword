"""
Pause injection for punctuation-based silence.

Converts punctuation marks to pause markers [Xs], then generates audio
with digital silence at those positions.
"""

import re
import torch
import logging
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

PAUSE_RE = re.compile(r"\[(\d+(?:\.\d+)?)s\]")

# Order matters: longer patterns first to avoid partial replacements
_PUNCT_ORDER = ["...", "--", ";", ":", ".", "!", "?", ","]


def inject_pauses_for_punctuation(text: str, pause_map: Dict[str, float]) -> str:
    """Replace punctuation characters with [Xs] pause markers.

    Args:
        text: Raw text to process.
        pause_map: Dict mapping punctuation char to pause duration in seconds.
                  E.g. {'.': 0.3, ',': 0.12, '?': 0.4}

    Returns:
        Text with punctuation replaced by pause markers like [0.3s].

    Examples:
        >>> inject_pauses_for_punctuation("Hello, world.", {',': 0.1, '.': 0.2})
        'Hello[0.1s] world[0.2s]'
    """
    result = text
    for punct in _PUNCT_ORDER:
        if punct in pause_map and pause_map[punct] > 0:
            seconds = pause_map[punct]
            marker = f"[{seconds}s]"
            result = result.replace(punct, marker)
    return result


def parse_text_with_pauses(raw: str):
    """Parse text containing [Xs] pause markers into events.

    Returns:
        events: List of ("text", str) or ("pause", float) tuples
        pauses: List of pause metadata dicts with index, seconds, positions
    """
    events: List[Tuple[str, Any]] = []
    pauses: List[Dict[str, Any]] = []

    cursor = 0
    pause_index = 0

    for m in PAUSE_RE.finditer(raw):
        # Add preceding text chunk
        chunk = raw[cursor : m.start()]
        if chunk:
            events.append(("text", chunk))

        # Add pause
        seconds = float(m.group(1))
        events.append(("pause", seconds))
        pauses.append(
            {
                "index": pause_index,
                "seconds": seconds,
                "start_char": m.start(),
                "end_char": m.end(),
            }
        )
        pause_index += 1

        cursor = m.end()

    # Add trailing text
    tail = raw[cursor:]
    if tail:
        events.append(("text", tail))

    return events, pauses


def generate_audio_with_pauses(tts_model, voice_state, raw_text: str):
    """Generate audio with pauses at [Xs] markers.

    Splits text by pause markers, generates TTS audio for text segments,
    and inserts digital silence for pauses.

    Args:
        tts_model: TTSModel instance
        voice_state: Voice conditioning state dict from get_state_for_audio_prompt()
        raw_text: Text with [Xs] pause markers, e.g. "Hello[0.3s] world[0.2s]"

    Returns:
        audio: Tensor of shape [samples] with TTS audio + silence at pauses
        pauses: List of pause metadata from parsing
    """
    events, pauses = parse_text_with_pauses(raw_text)
    sr = int(getattr(tts_model, "sample_rate", 24000))
    pieces = []
    audio_dtype = None
    audio_device = None

    for kind, value in events:
        if kind == "text":
            chunk = str(value).strip()
            if not chunk:
                continue
            logger.debug(f"Generating TTS for: {repr(chunk[:50])}")
            audio = tts_model.generate_audio(voice_state, chunk)
            if audio_dtype is None:
                audio_dtype = audio.dtype
                audio_device = audio.device
            pieces.append(audio)
        elif kind == "pause":
            seconds = float(value)
            n = max(0, int(round(seconds * sr)))
            # If no audio has been generated yet, pick a reasonable default
            dtype = audio_dtype or torch.float32
            device = audio_device or torch.device("cpu")
            logger.debug(f"Inserting silence: {seconds:.2f}s ({n} samples)")
            pieces.append(torch.zeros(n, dtype=dtype, device=device))

    if not pieces:
        return torch.zeros(0, dtype=torch.float32), pauses
    return torch.cat(pieces), pauses
