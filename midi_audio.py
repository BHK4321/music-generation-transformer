"""Minimal MIDI-to-audio playback without third-party packages.

This module parses standard MIDI files (format 0 and 1 with ticks-per-beat
timing), synthesizes a simple waveform from note events, writes a temporary
WAV file, and plays it through Windows speakers.

The synthesis is intentionally simple: it is good enough for demos and for
turning generated MIDI into something audible, but it is not a full sampler.
"""

from __future__ import annotations

import array
import math
import os
import struct
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class MidiNote:
    pitch: int
    start_tick: int
    end_tick: int
    velocity: int
    channel: int


def _read_u16(data: bytes, offset: int) -> Tuple[int, int]:
    return struct.unpack_from(">H", data, offset)[0], offset + 2


def _read_u32(data: bytes, offset: int) -> Tuple[int, int]:
    return struct.unpack_from(">I", data, offset)[0], offset + 4


def _read_vlq(data: bytes, offset: int) -> Tuple[int, int]:
    value = 0
    while True:
        byte = data[offset]
        offset += 1
        value = (value << 7) | (byte & 0x7F)
        if (byte & 0x80) == 0:
            break
    return value, offset


def _parse_midi_track(track_data: bytes) -> Tuple[List[MidiNote], List[Tuple[int, int]]]:
    notes: List[MidiNote] = []
    tempo_changes: List[Tuple[int, int]] = []
    offset = 0
    tick = 0
    running_status: Optional[int] = None
    active_notes: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    while offset < len(track_data):
        delta, offset = _read_vlq(track_data, offset)
        tick += delta

        status = track_data[offset]
        if status & 0x80:
            offset += 1
            running_status = status
        else:
            if running_status is None:
                raise ValueError("Malformed MIDI track: running status missing")
            status = running_status

        if status == 0xFF:
            meta_type = track_data[offset]
            offset += 1
            length, offset = _read_vlq(track_data, offset)
            meta_data = track_data[offset:offset + length]
            offset += length

            if meta_type == 0x51 and length == 3:
                tempo_us_per_beat = int.from_bytes(meta_data, byteorder="big")
                tempo_changes.append((tick, tempo_us_per_beat))
            elif meta_type == 0x2F:
                break
            continue

        event_type = status & 0xF0
        channel = status & 0x0F

        if event_type in (0xC0, 0xD0):
            offset += 1
            continue

        data1 = track_data[offset]
        data2 = track_data[offset + 1]
        offset += 2

        if event_type == 0x90:
            pitch = data1
            velocity = data2
            key = (channel, pitch)
            if velocity > 0:
                active_notes.setdefault(key, []).append((tick, velocity))
            else:
                stack = active_notes.get(key)
                if stack:
                    start_tick, start_velocity = stack.pop(0)
                    notes.append(MidiNote(pitch=pitch, start_tick=start_tick, end_tick=tick, velocity=start_velocity, channel=channel))
        elif event_type == 0x80:
            pitch = data1
            key = (channel, pitch)
            stack = active_notes.get(key)
            if stack:
                start_tick, start_velocity = stack.pop(0)
                notes.append(MidiNote(pitch=pitch, start_tick=start_tick, end_tick=tick, velocity=start_velocity, channel=channel))

    return notes, tempo_changes


def parse_midi_file(midi_path: str | Path) -> Tuple[int, List[MidiNote], List[Tuple[int, int]]]:
    """Parse a standard MIDI file into note events.

    Returns:
        ticks_per_beat, notes, tempo_changes
    """

    path = Path(midi_path)
    data = path.read_bytes()
    offset = 0

    if data[offset:offset + 4] != b"MThd":
        raise ValueError("Not a valid MIDI file: missing MThd header")
    offset += 4
    header_length, offset = _read_u32(data, offset)
    if header_length < 6:
        raise ValueError("Invalid MIDI header length")

    format_type, offset = _read_u16(data, offset)
    num_tracks, offset = _read_u16(data, offset)
    division, offset = _read_u16(data, offset)
    offset = 8 + header_length  # skip any extra header bytes

    if division & 0x8000:
        raise ValueError("SMPTE-timed MIDI files are not supported")

    ticks_per_beat = division
    all_notes: List[MidiNote] = []
    all_tempos: List[Tuple[int, int]] = []

    for _ in range(num_tracks):
        if data[offset:offset + 4] != b"MTrk":
            raise ValueError("Invalid MIDI file: missing MTrk chunk")
        offset += 4
        track_length, offset = _read_u32(data, offset)
        track_data = data[offset:offset + track_length]
        offset += track_length

        notes, tempos = _parse_midi_track(track_data)
        all_notes.extend(notes)
        all_tempos.extend(tempos)

    if not all_tempos:
        all_tempos = [(0, 500000)]  # default 120 BPM

    all_tempos.sort(key=lambda item: item[0])
    if all_tempos[0][0] != 0:
        all_tempos.insert(0, (0, 500000))

    return ticks_per_beat, all_notes, all_tempos


def _build_tick_to_second_converter(ticks_per_beat: int, tempo_changes: List[Tuple[int, int]]):
    tempo_changes = sorted(tempo_changes, key=lambda item: item[0])
    if not tempo_changes or tempo_changes[0][0] != 0:
        tempo_changes = [(0, 500000)] + tempo_changes

    def tick_to_seconds(tick: int) -> float:
        elapsed_seconds = 0.0
        current_tempo = tempo_changes[0][1]

        for index, (start_tick, tempo_us_per_beat) in enumerate(tempo_changes):
            if tick < start_tick:
                break
            current_tempo = tempo_us_per_beat
            if index + 1 < len(tempo_changes):
                next_tick = tempo_changes[index + 1][0]
                if tick < next_tick:
                    return elapsed_seconds + ((tick - start_tick) * current_tempo) / (ticks_per_beat * 1_000_000.0)
                elapsed_seconds += ((next_tick - start_tick) * current_tempo) / (ticks_per_beat * 1_000_000.0)
            else:
                return elapsed_seconds + ((tick - start_tick) * current_tempo) / (ticks_per_beat * 1_000_000.0)

        return elapsed_seconds

    return tick_to_seconds


def synthesize_notes_to_wav(
    notes: Iterable[MidiNote],
    ticks_per_beat: int,
    tempo_changes: List[Tuple[int, int]],
    wav_path: str | Path,
    sample_rate: int = 44_100,
    volume: float = 0.35,
) -> Path:
    """Render MIDI notes to a mono WAV file using a simple sine synth."""

    tick_to_seconds = _build_tick_to_second_converter(ticks_per_beat, tempo_changes)

    rendered_notes = []
    max_end = 0.0
    for note in notes:
        start_sec = tick_to_seconds(note.start_tick)
        end_sec = tick_to_seconds(note.end_tick)
        if end_sec <= start_sec:
            continue
        rendered_notes.append((note.pitch, start_sec, end_sec, note.velocity))
        max_end = max(max_end, end_sec)

    total_duration = max_end + 0.25
    total_samples = max(1, int(total_duration * sample_rate) + 1)
    buffer = [0.0] * total_samples

    attack = 0.01
    release = 0.04

    for pitch, start_sec, end_sec, velocity in rendered_notes:
        frequency = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        start_sample = max(0, int(start_sec * sample_rate))
        end_sample = min(total_samples, int(end_sec * sample_rate) + 1)
        velocity_scale = max(0.05, min(1.0, velocity / 127.0))

        for sample_index in range(start_sample, end_sample):
            t = sample_index / sample_rate
            if t < start_sec:
                continue
            local_t = t - start_sec
            if local_t < attack:
                envelope = local_t / attack
            elif (end_sec - t) < release:
                envelope = max(0.0, (end_sec - t) / release)
            else:
                envelope = 1.0

            # A slightly richer tone than a pure sine wave.
            phase = 2.0 * math.pi * frequency * local_t
            sample_value = (
                math.sin(phase)
                + 0.25 * math.sin(2.0 * phase)
                + 0.10 * math.sin(3.0 * phase)
            )
            buffer[sample_index] += volume * velocity_scale * envelope * sample_value

    peak = max((abs(value) for value in buffer), default=1.0)
    normalize = 0.98 / peak if peak > 0 else 1.0

    pcm = array.array("h")
    for value in buffer:
        clamped = max(-1.0, min(1.0, value * normalize))
        pcm.append(int(clamped * 32767))

    out_path = Path(wav_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())

    return out_path


def play_wav_file(wav_path: str | Path) -> None:
    """Play a WAV file through the default Windows audio device."""

    import winsound

    winsound.PlaySound(str(Path(wav_path)), winsound.SND_FILENAME)


def play_midi_file(midi_path: str | Path, keep_wav: bool = False) -> Path:
    """Convert a MIDI file to WAV and play it through the speakers.

    Returns the path to the rendered WAV file.
    """

    ticks_per_beat, notes, tempo_changes = parse_midi_file(midi_path)

    midi_path = Path(midi_path)
    if keep_wav:
        wav_path = midi_path.with_suffix(".wav")
        synthesize_notes_to_wav(notes, ticks_per_beat, tempo_changes, wav_path)
        play_wav_file(wav_path)
        return wav_path

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        synthesize_notes_to_wav(notes, ticks_per_beat, tempo_changes, tmp_path)
        play_wav_file(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return tmp_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert a MIDI file to audible audio and play it.")
    parser.add_argument("midi_file", type=str, help="Path to a .mid/.midi file")
    parser.add_argument("--keep-wav", action="store_true", help="Keep the rendered WAV file next to the MIDI file")
    args = parser.parse_args()

    out_wav = play_midi_file(args.midi_file, keep_wav=args.keep_wav)
    print(f"Rendered and played: {out_wav}")