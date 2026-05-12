"""MIDI event tokenization utilities for training a decoder-only Transformer.

This module turns standard MIDI note events into a compact token stream that
can be learned with next-token prediction, and it can decode generated tokens
back into notes for playback.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence

from midi_audio import MidiNote, parse_midi_file, synthesize_notes_to_wav


PIECE_START = "PIECE_START"
PIECE_END = "PIECE_END"
TIME_SHIFT_PREFIX = "TIME_SHIFT_"
NOTE_ON_PREFIX = "NOTE_ON_"
NOTE_OFF_PREFIX = "NOTE_OFF_"


def velocity_to_bucket(velocity: int, num_buckets: int = 8) -> int:
    velocity = max(1, min(127, int(velocity)))
    bucket = (velocity - 1) * num_buckets // 127 + 1
    return max(1, min(num_buckets, bucket))


def bucket_to_velocity(bucket: int, num_buckets: int = 8) -> int:
    bucket = max(1, min(num_buckets, int(bucket)))
    low = 1 + (bucket - 1) * 127 // num_buckets
    high = max(low, bucket * 127 // num_buckets)
    return int((low + high) // 2)


def _quantize_step(tick: int, grid_ticks: int) -> int:
    return max(0, int(round(tick / max(1, grid_ticks))))


def build_fixed_midi_vocab(
    max_time_shift: int = 64,
    num_velocity_buckets: int = 16,
) -> List[str]:
    """
    Build a fixed, universal MIDI vocabulary that covers all possible tokens.
    This ensures checkpoint compatibility across different datasets.
    
    Args:
        max_time_shift: Maximum time shift value (e.g., 64)
        num_velocity_buckets: Number of velocity buckets (e.g., 16)
    
    Returns:
        List of all possible MIDI tokens in order.
    """
    vocab = []
    
    # Special tokens for padding/control
    vocab.extend(["<pad>", "<unk>", "<bos>", "<eos>"])
    
    # Piece delimiters
    vocab.append(PIECE_START)
    vocab.append(PIECE_END)
    
    # NOTE_ON: all combinations of pitch (0-127) and velocity bucket (1-num_buckets)
    for pitch in range(128):
        for velocity_bucket in range(1, num_velocity_buckets + 1):
            vocab.append(f"{NOTE_ON_PREFIX}{pitch}_{velocity_bucket}")
    
    # NOTE_OFF: all pitches (0-127)
    for pitch in range(128):
        vocab.append(f"{NOTE_OFF_PREFIX}{pitch}")
    
    # TIME_SHIFT: 0 to max_time_shift
    for shift in range(max_time_shift + 1):
        vocab.append(f"{TIME_SHIFT_PREFIX}{shift}")
    
    return vocab


def save_vocab(vocab: List[str], output_path: str | Path) -> Path:
    """Save vocabulary to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(vocab, f)
    return output_path


def load_vocab(vocab_path: str | Path) -> List[str]:
    """Load vocabulary from a JSON file."""
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    return vocab


def encode_tokens_to_ids(tokens: Sequence[str], vocab: List[str]) -> List[int]:
    """Convert token strings to token IDs using the vocabulary."""
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    token_ids = []
    for token in tokens:
        token_ids.append(token_to_id.get(token, token_to_id.get("<unk>", 1)))
    return token_ids


def decode_token_ids_to_tokens(token_ids: Sequence[int], vocab: List[str]) -> List[str]:
    """Convert token IDs back to token strings using the vocabulary."""
    tokens = []
    for token_id in token_ids:
        if 0 <= int(token_id) < len(vocab):
            tokens.append(vocab[int(token_id)])
        else:
            tokens.append("<unk>")
    return tokens


def midi_file_to_event_tokens(
    midi_path: str | Path,
    grid_ticks: int = 120,
    max_time_shift: int = 64,
    num_velocity_buckets: int = 8,
) -> List[str]:
    """Convert one MIDI file into a sequence of event tokens."""

    ticks_per_beat, notes, _tempo_changes = parse_midi_file(midi_path)
    if not notes:
        return [PIECE_START, PIECE_END]

    # Normalize to a fixed grid so files with different PPQ values still map
    # to the same event vocabulary.
    scale = max(1, ticks_per_beat // max(1, grid_ticks))

    events: List[tuple[int, int, str]] = []
    for note in notes:
        start_step = _quantize_step(note.start_tick // scale, 1)
        end_step = _quantize_step(note.end_tick // scale, 1)
        if end_step <= start_step:
            end_step = start_step + 1
        pitch = max(0, min(127, int(note.pitch)))
        velocity_bucket = velocity_to_bucket(note.velocity, num_buckets=num_velocity_buckets)
        events.append((start_step, 1, f"{NOTE_ON_PREFIX}{pitch}_{velocity_bucket}"))
        events.append((end_step, 0, f"{NOTE_OFF_PREFIX}{pitch}"))

    events.sort(key=lambda item: (item[0], item[1]))

    tokens = [PIECE_START]
    current_step = 0
    for event_step, _order, token in events:
        delta = max(0, event_step - current_step)
        while delta > 0:
            shift = min(delta, max_time_shift)
            tokens.append(f"{TIME_SHIFT_PREFIX}{shift}")
            current_step += shift
            delta -= shift
        tokens.append(token)

    tokens.append(PIECE_END)
    return tokens


def midi_files_to_corpus(
    midi_files: Sequence[str | Path],
    limit: int | None = None,
    grid_ticks: int = 120,
    max_time_shift: int = 64,
    num_velocity_buckets: int = 8,
) -> List[str]:
    """Convert many MIDI files into a single token corpus."""

    corpus: List[str] = []
    for index, midi_file in enumerate(midi_files):
        if limit is not None and index >= limit:
            break
        corpus.extend(
            midi_file_to_event_tokens(
                midi_file,
                grid_ticks=grid_ticks,
                max_time_shift=max_time_shift,
                num_velocity_buckets=num_velocity_buckets,
            )
        )
    if not corpus:
        raise ValueError("No MIDI tokens were produced")
    return corpus


def event_tokens_to_notes(
    tokens: Iterable[str],
    step_ticks: int = 120,
    num_velocity_buckets: int = 8,
) -> List[MidiNote]:
    """Decode event tokens into note objects suitable for rendering."""

    current_tick = 0
    active_notes: dict[int, list[tuple[int, int]]] = defaultdict(list)
    notes: List[MidiNote] = []

    for token in tokens:
        if token in {PIECE_START, PIECE_END}:
            continue
        if token.startswith(TIME_SHIFT_PREFIX):
            shift = int(token[len(TIME_SHIFT_PREFIX):])
            # TIME_SHIFT_n means advance by n token-time steps.
            current_tick += max(0, shift)
            continue
        if token.startswith(NOTE_ON_PREFIX):
            payload = token[len(NOTE_ON_PREFIX):]
            pitch_str, bucket_str = payload.split("_")
            pitch = int(pitch_str)
            velocity = bucket_to_velocity(int(bucket_str), num_buckets=num_velocity_buckets)
            active_notes[pitch].append((current_tick, velocity))
            continue
        if token.startswith(NOTE_OFF_PREFIX):
            pitch = int(token[len(NOTE_OFF_PREFIX):])
            stack = active_notes.get(pitch)
            if stack:
                start_tick, velocity = stack.pop(0)
                if current_tick <= start_tick:
                    current_tick = start_tick + step_ticks
                notes.append(MidiNote(pitch=pitch, start_tick=start_tick, end_tick=current_tick, velocity=velocity, channel=0))

    for pitch, stack in active_notes.items():
        while stack:
            start_tick, velocity = stack.pop(0)
            notes.append(MidiNote(pitch=pitch, start_tick=start_tick, end_tick=start_tick + 1, velocity=velocity, channel=0))

    notes.sort(key=lambda note: (note.start_tick, note.pitch, note.end_tick))
    return notes


def render_event_tokens_to_wav(
    tokens: Sequence[str],
    wav_path: str | Path,
    step_seconds: float = 0.12,
    step_ticks: int = 100,
) -> Path:
    """Render event tokens directly into a WAV file and return its path."""

    notes = event_tokens_to_notes(tokens, step_ticks=step_ticks)
    tempo_us_per_beat = int(step_seconds * 1_000_000.0 * step_ticks)
    return synthesize_notes_to_wav(
        notes,
        ticks_per_beat=step_ticks,
        tempo_changes=[(0, tempo_us_per_beat)],
        wav_path=wav_path,
    )


def render_token_ids_to_wav(
    token_ids: Sequence[int],
    vocab: Sequence[str],
    wav_path: str | Path,
    step_seconds: float = 0.12,
    step_ticks: int = 100,
) -> Path:
    """Decode token IDs, render them to WAV, and return the WAV path."""

    tokens = decode_token_ids_to_tokens(token_ids, vocab)
    return render_event_tokens_to_wav(tokens, wav_path, step_seconds=step_seconds, step_ticks=step_ticks)