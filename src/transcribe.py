from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import soundfile as sf

from src import config as cfg


class TranscribedNote(NamedTuple):

    pitch: int  # MIDI note number
    start: float
    end: float
    amplitude: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class TranscriptionParams:

    onset_threshold: float = cfg.ONSET_THRESHOLD
    frame_threshold: float = cfg.FRAME_THRESHOLD
    minimum_note_length_ms: float = cfg.MINIMUM_NOTE_LENGTH_MS

    def fingerprint(self) -> str:
        payload = (
            f"{self.onset_threshold}|{self.frame_threshold}|"
            f"{self.minimum_note_length_ms}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def as_dict(self) -> dict[str, float]:
        return {
            "onset_threshold": self.onset_threshold,
            "frame_threshold": self.frame_threshold,
            "minimum_note_length_ms": self.minimum_note_length_ms,
        }


@dataclass(frozen=True)
class Transcription:
    notes: tuple[TranscribedNote, ...]
    params: TranscriptionParams
    source_path: Path
    from_cache: bool = False

    @property
    def note_count(self) -> int:
        return len(self.notes)


def _audio_digest(path: Path) -> str:
    """Digest of the decoded samples, not of the file's bytes.

    libsndfile stamps the time into every float WAV it writes (its PEAK
    chunk), so the same audio prepared again in a later run never had the
    same bytes, and the cache never hit across runs.
    """
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    digest = hashlib.sha256(str(sample_rate).encode())
    digest.update(np.ascontiguousarray(samples).tobytes())
    return digest.hexdigest()[:16]


def _sorted_notes(raw: Sequence[tuple]) -> tuple[TranscribedNote, ...]:

    notes = [
        TranscribedNote(
            pitch=int(pitch),
            start=float(start),
            end=float(end),
            amplitude=float(amplitude),
        )
        for start, end, pitch, amplitude, *_ in raw
    ]
    notes.sort(key=lambda note: (note.start, note.pitch, note.end))
    return tuple(notes)


def transcribe(
    audio_path: str | Path,
    params: TranscriptionParams | None = None,
    midi_path: str | Path | None = None,
) -> Transcription:

    audio_path = Path(audio_path)
    params = params or TranscriptionParams()

    # imported here, not at module scope: basic-pitch pulls in TensorFlow,
    # which costs seconds and prints banners
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    _, midi_data, note_events = predict(
        audio_path,
        ICASSP_2022_MODEL_PATH,
        onset_threshold=params.onset_threshold,
        frame_threshold=params.frame_threshold,
        minimum_note_length=params.minimum_note_length_ms,
    )

    if midi_path is not None:
        midi_path = Path(midi_path)
        midi_path.parent.mkdir(parents=True, exist_ok=True)
        midi_data.write(str(midi_path))

    return Transcription(
        notes=_sorted_notes(note_events),
        params=params,
        source_path=audio_path,
    )


def transcribe_cached(
    audio_path: str | Path,
    cache_dir: str | Path,
    params: TranscriptionParams | None = None,
    midi_path: str | Path | None = None,
) -> Transcription:

    audio_path = Path(audio_path)
    params = params or TranscriptionParams()

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / (
        f"{audio_path.stem}.{_audio_digest(audio_path)}.{params.fingerprint()}.json"
    )

    if cache_file.is_file():
        stored = json.loads(cache_file.read_text())
        return Transcription(
            notes=tuple(TranscribedNote(*note) for note in stored["notes"]),
            params=params,
            source_path=audio_path,
            from_cache=True,
        )

    result = transcribe(audio_path, params, midi_path)
    # written aside and then renamed, so a run stopped with Ctrl+C mid-write
    # leaves no half-written entry for the next run to choke on
    partial = cache_file.with_suffix(".partial")
    partial.write_text(
        json.dumps(
            {
                "source": audio_path.name,
                "params": params.as_dict(),
                "notes": [list(note) for note in result.notes],
            },
            indent=1,
        )
    )
    partial.replace(cache_file)
    return result


def describe(transcription: Transcription) -> dict[str, float | int | str]:
    notes = transcription.notes
    if not notes:
        return {"file": transcription.source_path.name, "notes": 0}
    durations = [note.duration for note in notes]
    return {
        "file": transcription.source_path.name,
        "notes": len(notes),
        "pitch_min": min(note.pitch for note in notes),
        "pitch_max": max(note.pitch for note in notes),
        "shortest_note_ms": round(min(durations) * 1000, 2),
        "median_note_ms": round(sorted(durations)[len(durations) // 2] * 1000, 2),
        "span_s": round(max(note.end for note in notes), 3),
    }
