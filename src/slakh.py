"""Turn the Slakh stem subset into a corpus the pipeline can actually encode.

Slakh gives exact ground truth: the audio is rendered from the MIDI, so onsets
and pitches are known without annotation. What it does not give is material that
fits the token scheme. Three things have to be settled first.

Tempo. The header carries one 8-bit tempo and nothing else, so a track whose
MIDI contains a tempo map cannot be represented in one piece. Flattening it to
its first tempo would put every later onset on the wrong grid, so it is cut at
each tempo change instead, and every stretch of one tempo is encoded with that
tempo in its own header.

Long rests. A delta is 8 bits, so at most 255 grid units may separate two
onsets, and `notes_to_tokens` seeds the previous position at zero, which makes
the first note's delta its absolute position. A stem that sits out a section, or
that does not enter until late, exceeds that. Since the error handling is gone
the value would wrap silently and still round-trip losslessly, so the corruption
would surface as a lower F-measure blamed on quantisation. Such tracks are cut
at their long rests into segments instead, which is what the music is doing
anyway.

Durations. `quantize` already clamps to 1..64 units and counts it, so a held
note degrades measurably rather than wrapping. Nothing to do but report it.

Audibility. The audio is rendered from the MIDI, but through sampled patches
that are silent outside their range, and the MIDI keeps every note regardless.
A segment whose reference is largely notes nobody can hear would measure the
rendering rather than the transcription, so it is dropped.

    python -m src.slakh --source new_dataset/slakh_subset --out data/slakh
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

from src import config as cfg
from src.quantize import quantize
from src.tokenizer import compress, notes_to_tokens

# a segment shorter than this is a fragment, not a phrase worth measuring
MIN_SEGMENT_NOTES = 32

# silence kept either side of a cut, so a segment never opens on its own onset
SEGMENT_PAD_S = 1.0

# a note this close before a tempo change is taken to start on it
ONSET_EPS_S = 1e-6

# a reference note whose attack stays below SILENCE_DBFS was never rendered;
# a segment with more than MAX_SILENT_SHARE of such notes is dropped
SILENCE_DBFS = -60.0
ATTACK_WINDOW_S = 0.1
MAX_SILENT_SHARE = 0.10

MANIFEST_COLUMNS = (
    "sample",
    "tempo",
    "grid",
    "durasi",
    "kerapatan",
    "pengulangan",
    "split",
    "inst_class",
    "source_track",
    "segment",
    "notes",
    "span_s",
    "audio_path",
    "midi_path",
)

DURATION_LEVELS = ("pendek", "sedang", "panjang")
MAGNITUDE_LEVELS = ("rendah", "sedang", "tinggi")


@dataclass(frozen=True)
class StemSource:
    """One stem a track directory carries; a track may carry several."""

    track: str
    stem_id: str
    audio_path: Path
    midi_path: Path
    inst_class: str
    program_name: str
    is_drum: bool

    @property
    def name(self) -> str:
        return f"{self.track}_{self.stem_id}"


@dataclass
class Segment:
    """A stretch of one stem that fits the token scheme."""

    source: StemSource
    index: int
    start_s: float
    end_s: float
    tempo: int
    grid_code: int
    notes: list  # pretty_midi.Note, times still on the original timeline
    grid_error_ratio: float = 0.0
    span_s: float = 0.0
    density: float = 0.0
    repetition_pct: float = 0.0
    clamped_durations: int = 0
    truncated_durations: int = 0
    max_delta: int = 0

    @property
    def name(self) -> str:
        if self.index < 0:
            return f"{self.source.track}_{self.source.stem_id}"
        return f"{self.source.track}_{self.source.stem_id}_s{self.index:02d}"

    @property
    def is_whole_track(self) -> bool:
        return self.index < 0


@dataclass
class Rejection:
    track: str
    reason: str
    detail: str


@dataclass(frozen=True)
class TempoRegion:
    """A stretch of a stem played at one whole-BPM tempo."""

    start_s: float  # where the tempo takes effect
    end_s: float  # where the next one does
    bpm: int
    origin_s: float  # a line of every candidate grid, at or before start_s


@dataclass
class Report:
    segments: list[Segment] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    scanned: int = 0
    segmented_tracks: int = 0
    tempo_map_stems: int = 0
    tempo_map_notes: int = 0
    tempo_map_notes_kept: int = 0


def _stem_metadata(track_dir: Path, stem_id: str) -> dict[str, str]:
    """Pull one stem's block out of metadata.yaml, every field as a string.

    The file lists every stem of the original Slakh track, but the subset may
    ship only some of them, so the block is picked by id rather than taken
    wholesale. Parsed with a regex to avoid a PyYAML dependency for a flat block.
    """
    text = (track_dir / "metadata.yaml").read_text(encoding="utf-8", errors="replace")
    block = re.search(r"^  %s:\n((?:    .*\n)+)" % re.escape(stem_id), text, re.M)
    if block is None:
        return {}
    return {
        key: value.strip()
        for key, value in re.findall(r"^    (\w+): (.*)$", block.group(1), re.M)
    }


def scan(source: str | Path) -> list[StemSource]:
    """Every stem with both audio and MIDI, in every track directory."""
    source = Path(source)
    stems: list[StemSource] = []

    for track_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        for audio_path in sorted((track_dir / "stems").glob("*.flac")):
            stem_id = audio_path.stem
            midi_path = track_dir / "MIDI" / f"{stem_id}.mid"
            if not midi_path.is_file():
                continue

            meta = _stem_metadata(track_dir, stem_id)
            stems.append(
                StemSource(
                    track=track_dir.name,
                    stem_id=stem_id,
                    audio_path=audio_path,
                    midi_path=midi_path,
                    inst_class=meta.get("inst_class", "Unknown"),
                    program_name=meta.get("midi_program_name", "Unknown"),
                    is_drum=meta.get("is_drum", "false") == "true",
                )
            )

    return stems


def _constant_tempo(midi: pretty_midi.PrettyMIDI) -> int | None:
    """The single tempo of the piece, or None if it changes.

    Lakh MIDI often repeats the same tempo event several times, which is not a
    tempo map; only distinct integer BPMs count as a change, since the header
    stores whole BPM anyway.
    """
    _, tempi = midi.get_tempo_changes()
    if len(tempi) == 0:
        return None
    rounded = {int(round(float(value))) for value in tempi}
    if len(rounded) != 1:
        return None
    return rounded.pop()


def _tempo_regions(midi: pretty_midi.PrettyMIDI) -> list[TempoRegion]:
    """The stretches of one whole BPM a tempo map divides the piece into.

    Consecutive events that round to the same BPM are one stretch, as in
    `_constant_tempo`. Each stretch also gets the origin its grid is measured
    from. The grid lines of the piece sit at fixed tick positions counted from
    the start, so a change that falls between them leaves the new stretch's
    lines out of phase with the change itself. Every candidate of `_fit_grid`
    has a line on each half note (a quarter at half the tempo), so the origin is
    the last half-note tick at or before the change, placed in time at the new
    tempo.
    """
    times, tempi = midi.get_tempo_changes()
    marks: list[tuple[float, float]] = []
    for time, tempo in zip(times, tempi):
        if marks and int(round(marks[-1][1])) == int(round(float(tempo))):
            continue
        marks.append((float(time), float(tempo)))

    half_note = 2 * midi.resolution
    regions = []
    for index, (start, tempo) in enumerate(marks):
        behind = midi.time_to_tick(start) % half_note
        regions.append(
            TempoRegion(
                start_s=start,
                end_s=marks[index + 1][0] if index + 1 < len(marks) else math.inf,
                bpm=int(round(tempo)),
                origin_s=start - behind * 60.0 / (tempo * midi.resolution),
            )
        )
    return regions


def _shifted(notes: list, offset_s: float) -> list:
    """Copies of `notes` with `offset_s` taken off every time."""
    return [
        pretty_midi.Note(
            velocity=note.velocity,
            pitch=note.pitch,
            start=note.start - offset_s,
            end=note.end - offset_s,
        )
        for note in notes
    ]


def _clipped(notes: list, end_s: float) -> list:
    """`notes` with any note held past `end_s` cut off there, as its audio is."""
    return [
        note
        if note.end <= end_s
        else pretty_midi.Note(
            velocity=note.velocity,
            pitch=note.pitch,
            start=note.start,
            end=max(end_s, note.start + 1e-3),
        )
        for note in notes
    ]


#: how far off the grid the reference may sit before the grid model is wrong,
#: as a fraction of one grid unit; a quarter is the tolerance corpus.py applies
MAX_GRID_ERROR_RATIO = 0.25


def _fit_grid(notes: list, notated_tempo: int) -> tuple[int, int, float]:
    """Pick the tempo and grid resolution the piece actually sits on.

    Both are header fields -- 8 bits of tempo and 2 of grid -- so choosing them
    per sample is the format working as designed, not a change to it. Two things
    make the notated value unusable as given. Lakh MIDI routinely notates a
    piece at half or double its perceived tempo, which puts every offbeat note
    between grid lines; and a piece in triplets needs six subdivisions where one
    in sixteenths needs four, so no single grid fits the corpus.

    Returns the pair with the smallest RMS deviation from the grid, measured in
    units of the grid itself so the resolutions compare fairly.
    """
    best = (notated_tempo, cfg.DEFAULT_GRID_CODE, float("inf"))
    for multiplier in (0.5, 1.0, 2.0):
        tempo = int(round(notated_tempo * multiplier))
        if not cfg.MIN_TEMPO <= tempo <= cfg.MAX_TEMPO:
            continue
        for grid_code in sorted(cfg.GRID_SUBDIVISIONS):
            report = quantize(notes, tempo, grid_code)
            ratio = report.rms_grid_error_ms / report.grid_unit_ms
            if ratio < best[2]:
                best = (tempo, grid_code, ratio)
    return best


def _split_points(
    notes: list, tempo: int, grid_code: int
) -> list[int]:
    """Indices where a rest is too long for an 8-bit delta to cross.

    Index 0 counts: the first delta is measured from position zero, so a late
    entry overflows exactly like a mid-piece rest.
    """
    delta_s = cfg.grid_unit_seconds(tempo, grid_code)
    limit_s = cfg.MAX_DELTA_UNITS * delta_s

    cuts: list[int] = []
    previous_onset = 0.0
    for index, note in enumerate(notes):
        if note.start - previous_onset > limit_s:
            cuts.append(index)
        previous_onset = note.start
    return cuts


def _measure(segment: Segment) -> None:
    """Fill in what the segment contains, on its own shifted timeline."""
    shifted = _shifted(segment.notes, segment.start_s)

    report = quantize(shifted, segment.tempo, segment.grid_code)
    tokens = notes_to_tokens(report.notes)

    segment.clamped_durations = report.clamped_count
    segment.truncated_durations = report.truncated_count
    segment.grid_error_ratio = report.rms_grid_error_ms / report.grid_unit_ms
    segment.max_delta = max((token.delta for token in tokens), default=0)
    segment.span_s = max(note.end for note in shifted) - min(
        note.start for note in shifted
    )
    segment.density = len(shifted) / segment.span_s if segment.span_s > 0 else 0.0
    segment.repetition_pct = (
        round((1 - len(compress(tokens)) / len(tokens)) * 100.0, 2) if tokens else 0.0
    )


def _segments_for(
    stem: StemSource, grid_code: int, report: Report
) -> list[Segment]:
    """Cut one stem into pieces the token scheme can hold, or reject it."""
    midi = pretty_midi.PrettyMIDI(str(stem.midi_path))

    if stem.is_drum:
        report.rejections.append(
            Rejection(stem.name, "drums", "percussion pitches are not pitches")
        )
        return []

    notes = sorted(
        (note for inst in midi.instruments for note in inst.notes),
        key=lambda note: (note.start, note.pitch),
    )
    if not notes:
        report.rejections.append(Rejection(stem.name, "empty", "no notes in the stem"))
        return []

    audio_info = sf.info(str(stem.audio_path))
    audio_duration = audio_info.frames / audio_info.samplerate

    notated = _constant_tempo(midi)
    if notated is None:
        return _tempo_map_segments(stem, midi, notes, audio_duration, report)

    if not cfg.MIN_TEMPO <= notated <= cfg.MAX_TEMPO:
        report.rejections.append(
            Rejection(
                stem.name,
                "tempo_range",
                f"{notated} BPM is outside {cfg.MIN_TEMPO}..{cfg.MAX_TEMPO}",
            )
        )
        return []

    # before anything else, since the chosen tempo sets the grid unit and so
    # decides where the delta limit falls
    tempo, grid_code, ratio = _fit_grid(notes, notated)
    if ratio > MAX_GRID_ERROR_RATIO:
        report.rejections.append(
            Rejection(
                stem.name,
                "off_grid",
                f"best fit still {ratio:.2f} of a grid unit off at {tempo} BPM, "
                "grid %d; the piece is played rather than sequenced" % grid_code,
            )
        )
        return []

    cuts = _split_points(notes, tempo, grid_code)
    if not cuts:
        # nothing to cut, so the original files can be used as they are
        whole = Segment(stem, -1, 0.0, audio_duration, tempo, grid_code, notes)
        _measure(whole)
        if len(notes) < MIN_SEGMENT_NOTES:
            report.rejections.append(
                Rejection(stem.name, "too_short", f"{len(notes)} notes")
            )
            return []
        return [whole]

    report.segmented_tracks += 1

    bounds = [0] + cuts + [len(notes)]
    segments: list[Segment] = []
    _cut(stem, notes, list(zip(bounds, bounds[1:])), tempo, grid_code, 0.0,
         audio_duration, segments, report)

    if not segments:
        report.rejections.append(
            Rejection(stem.name, "too_short", "no segment reached the note floor")
        )
    return segments


def _tempo_map_segments(
    stem: StemSource,
    midi: pretty_midi.PrettyMIDI,
    notes: list,
    audio_duration: float,
    report: Report,
) -> list[Segment]:
    """Cut a stem whose tempo changes into its stretches of one tempo.

    Each stretch is fitted and cut exactly as a whole stem of constant tempo
    is, on a grid measured from the stretch's own origin. A stretch too short to
    hold a segment is dropped; ritardandi written as a run of small tempo steps
    end up there, since each step holds only a few notes.
    """
    regions = _tempo_regions(midi)
    report.tempo_map_stems += 1
    report.tempo_map_notes += len(notes)

    segments: list[Segment] = []
    lo = 0
    for number, region in enumerate(regions):
        hi = lo
        while hi < len(notes) and notes[hi].start < region.end_s - ONSET_EPS_S:
            hi += 1
        first, lo = lo, hi
        group = notes[first:hi]
        if len(group) < MIN_SEGMENT_NOTES:
            continue

        where = f"tempo region {number} ({region.bpm} BPM from {region.start_s:.1f} s)"
        if not cfg.MIN_TEMPO <= region.bpm <= cfg.MAX_TEMPO:
            report.rejections.append(
                Rejection(
                    stem.name,
                    "tempo_range",
                    f"{where} is outside {cfg.MIN_TEMPO}..{cfg.MAX_TEMPO}",
                )
            )
            continue

        local = _shifted(group, region.origin_s)
        tempo, grid_code, ratio = _fit_grid(local, region.bpm)
        if ratio > MAX_GRID_ERROR_RATIO:
            report.rejections.append(
                Rejection(
                    stem.name,
                    "off_grid",
                    f"{where}: best fit still {ratio:.2f} of a grid unit off",
                )
            )
            continue

        bounds = [0] + _split_points(local, tempo, grid_code) + [len(group)]
        ranges = [(first + a, first + b) for a, b in zip(bounds, bounds[1:])]
        _cut(stem, notes, ranges, tempo, grid_code, region.origin_s,
             audio_duration, segments, report)

    report.tempo_map_notes_kept += sum(len(segment.notes) for segment in segments)
    if not segments:
        report.rejections.append(
            Rejection(
                stem.name,
                "tempo_map",
                f"none of {len(regions)} tempo regions yields "
                f"{MIN_SEGMENT_NOTES} notes on one grid",
            )
        )
    return segments


def _cut(
    stem: StemSource,
    notes: list,
    ranges: list[tuple[int, int]],
    tempo: int,
    grid_code: int,
    origin_s: float,
    audio_duration: float,
    segments: list[Segment],
    report: Report,
) -> None:
    """Turn index ranges of a stem's notes into segments, appending to `segments`
    the ones that still fit once cut. `origin_s` is where the grid is counted
    from: zero for a stem of one tempo, the region's origin otherwise.
    """
    grid_unit = cfg.grid_unit_seconds(tempo, grid_code)
    for lo, hi in ranges:
        group = notes[lo:hi]
        if len(group) < MIN_SEGMENT_NOTES:
            continue

        first_onset = group[0].start
        last_offset = max(note.end for note in group)
        # cut inside the rest, not on the note, so the slice keeps a run-up
        previous_offset = max((note.end for note in notes[:lo]), default=0.0)
        next_onset = notes[hi].start if hi < len(notes) else audio_duration

        # a note held across the cut leaves no rest, and no run-up either
        run_up = min(SEGMENT_PAD_S, max(0.0, first_onset - previous_offset))
        start_s = max(0.0, first_onset - run_up)
        end_s = min(audio_duration, last_offset + min(SEGMENT_PAD_S, next_onset - last_offset))

        # snap the cut down to a grid line. Shifting the segment to its own
        # timeline subtracts start_s from every onset, so a cut off the grid
        # moves the whole segment off it by that same phase and undoes the fit.
        start_s = origin_s + math.floor((start_s - origin_s) / grid_unit) * grid_unit
        if start_s < 0.0:
            start_s += grid_unit

        segment = Segment(
            stem, len(segments), start_s, end_s, tempo, grid_code, _clipped(group, end_s)
        )
        _measure(segment)
        # the fit was measured on the whole track; a single segment can still
        # drift, so it is checked again on its own notes
        if segment.grid_error_ratio > MAX_GRID_ERROR_RATIO:
            report.rejections.append(
                Rejection(
                    stem.name,
                    "off_grid",
                    f"segment {len(segments)} sits {segment.grid_error_ratio:.2f} "
                    "of a unit off after the cut",
                )
            )
            continue
        if segment.max_delta > cfg.MAX_DELTA_UNITS:
            report.rejections.append(
                Rejection(
                    stem.name,
                    "delta_overflow",
                    f"segment {len(segments)} still spans {segment.max_delta} units",
                )
            )
            continue
        segments.append(segment)


def _silent_share(segment: Segment) -> float:
    """Share of the segment's reference notes that start in silence.

    Slakh renders each stem through a sampled instrument, and a patch outside
    its range plays nothing: a bass trombone patch is silent above F#4, yet the
    MIDI keeps every note written for it. Such notes are in the reference but
    not in the audio, so no transcription can find them. A note counts as
    silent when the first 100 ms after its onset stay below -60 dBFS. One that
    is silent under another that sounds goes unnoticed, so this is a lower bound.
    """
    info = sf.info(str(segment.source.audio_path))
    data, rate = sf.read(
        str(segment.source.audio_path),
        start=int(round(segment.start_s * info.samplerate)),
        stop=int(round(segment.end_s * info.samplerate)),
        dtype="float32",
        always_2d=True,
    )
    level = np.abs(data).max(axis=1)
    floor = 10.0 ** (SILENCE_DBFS / 20.0)

    silent = 0
    for note in segment.notes:
        onset = note.start - segment.start_s
        window = min(ATTACK_WINDOW_S, max(note.end - note.start, 0.02))
        first, last = int(onset * rate), int((onset + window) * rate) + 1
        if first >= len(level) or level[first:last].max(initial=0.0) < floor:
            silent += 1
    return silent / len(segment.notes)


def _write_segment(segment: Segment, out_dir: Path) -> tuple[Path, Path]:
    """Materialise a cut segment as its own audio and MIDI pair.

    Whole tracks are referenced in place; only cut ones are rewritten, which
    keeps the prepared corpus small.
    """
    if segment.is_whole_track:
        return segment.source.audio_path, segment.source.midi_path

    audio_out = out_dir / "audio" / f"{segment.name}.flac"
    midi_out = out_dir / "ground_truth" / f"{segment.name}.mid"
    audio_out.parent.mkdir(parents=True, exist_ok=True)
    midi_out.parent.mkdir(parents=True, exist_ok=True)

    info = sf.info(str(segment.source.audio_path))
    start_frame = int(round(segment.start_s * info.samplerate))
    stop_frame = int(round(segment.end_s * info.samplerate))
    data, rate = sf.read(
        str(segment.source.audio_path), start=start_frame, stop=stop_frame, dtype="float32"
    )
    sf.write(str(audio_out), data, rate)

    written = pretty_midi.PrettyMIDI(initial_tempo=float(segment.tempo))
    instrument = pretty_midi.Instrument(program=0, name=segment.source.program_name)
    instrument.notes = _shifted(segment.notes, segment.start_s)
    written.instruments.append(instrument)
    written.write(str(midi_out))

    return audio_out, midi_out


def _tercile_levels(values: list[float], levels: tuple[str, str, str]) -> list[str]:
    """Label each value by which third of the corpus it falls in.

    Terciles rather than fixed thresholds because Tabel 3.4 asks for three
    populated levels, and what counts as dense depends on the corpus in hand.
    """
    ordered = sorted(values)
    if not ordered:
        return []
    low = ordered[len(ordered) // 3]
    high = ordered[2 * len(ordered) // 3]
    return [
        levels[0] if value < low else (levels[2] if value >= high else levels[1])
        for value in values
    ]


def _assign_splits(
    segments: list[Segment],
    dev_fraction: float,
    seed: int,
    fixed: dict[str, str] | None = None,
) -> list[str]:
    """Hold out a development set for calibration, split by track.

    The onset threshold has to be tuned somewhere other than where it is
    reported, and nothing from one track may straddle the split: not the
    segments cut from one stem, and not the other stems of the same song, which
    share its tempo and its repeated sections. Otherwise the dev set leaks into
    the test set.

    When every track carries a single instrument class, tracks are stratified
    by that class, which reproduces the one-stem corpus exactly. Once a track
    carries several classes there is no class to stratify it by, so tracks are
    drawn from one pool instead.

    Tracks named in `fixed` keep the split given there, and only the others
    are drawn, at the same fraction. Growing a corpus this way leaves the dev
    set the threshold was calibrated on in place, where redrawing every track
    would scatter it across both splits.
    """
    fixed = fixed or {}
    by_track: dict[str, list[int]] = {}
    for index, segment in enumerate(segments):
        by_track.setdefault(segment.source.track, []).append(index)

    classes_by_track = {
        track: {segments[index].source.inst_class for index in indices}
        for track, indices in by_track.items()
    }
    single_class = all(len(classes) == 1 for classes in classes_by_track.values())

    tracks_by_inst: dict[str, list[str]] = {}
    for track, classes in classes_by_track.items():
        if track in fixed:
            continue
        stratum = next(iter(classes)) if single_class else "all"
        tracks_by_inst.setdefault(stratum, []).append(track)

    splits = [fixed.get(segment.source.track, "test") for segment in segments]
    rng = random.Random(seed)
    for tracks in tracks_by_inst.values():
        tracks = sorted(tracks)
        rng.shuffle(tracks)
        for track in tracks[: max(1, round(len(tracks) * dev_fraction))]:
            for index in by_track[track]:
                splits[index] = "dev"
    return splits


def prepare(
    source: str | Path,
    out_dir: str | Path,
    grid_code: int = cfg.DEFAULT_GRID_CODE,
    dev_fraction: float = 0.2,
    seed: int = 0,
    fixed_splits: dict[str, str] | None = None,
) -> Report:
    """Build a corpus manifest, writing only the segments that had to be cut."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = Report()
    stems = scan(source)
    report.scanned = len(stems)
    for stem in stems:
        for segment in _segments_for(stem, grid_code, report):
            share = _silent_share(segment)
            if share > MAX_SILENT_SHARE:
                report.rejections.append(
                    Rejection(
                        segment.name,
                        "inaudible",
                        f"{share:.0%} of its reference notes start in silence",
                    )
                )
                continue
            report.segments.append(segment)

    segments = report.segments
    duration_levels = _tercile_levels([s.span_s for s in segments], DURATION_LEVELS)
    density_levels = _tercile_levels([s.density for s in segments], MAGNITUDE_LEVELS)
    repeat_levels = _tercile_levels([s.repetition_pct for s in segments], MAGNITUDE_LEVELS)
    splits = _assign_splits(segments, dev_fraction, seed, fixed_splits)

    rows = []
    for index, segment in enumerate(segments):
        audio_path, midi_path = _write_segment(segment, out_dir)
        rows.append(
            {
                "sample": segment.name,
                "tempo": segment.tempo,
                "grid": segment.grid_code,
                "durasi": duration_levels[index],
                "kerapatan": density_levels[index],
                "pengulangan": repeat_levels[index],
                "split": splits[index],
                "inst_class": segment.source.inst_class,
                "source_track": segment.source.track,
                "segment": "whole" if segment.is_whole_track else segment.index,
                "notes": len(segment.notes),
                "span_s": round(segment.span_s, 2),
                "audio_path": audio_path.as_posix(),
                "midi_path": midi_path.as_posix(),
            }
        )

    with open(out_dir / "corpus.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "preflight.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "grid_code": grid_code,
                "seed": seed,
                "tracks_with_kept_split": len(
                    {s.source.track for s in segments} & set(fixed_splits or {})
                ),
                "stems_scanned": report.scanned,
                "stems_segmented": report.segmented_tracks,
                "stems_with_tempo_map": report.tempo_map_stems,
                "tempo_map_notes": report.tempo_map_notes,
                "tempo_map_notes_kept": report.tempo_map_notes_kept,
                "segments_kept": len(segments),
                "tracks_kept": len({s.source.track for s in segments}),
                "dev": splits.count("dev"),
                "test": splits.count("test"),
                "rejections": [
                    {"track": r.track, "reason": r.reason, "detail": r.detail}
                    for r in report.rejections
                ],
                "notes_total": sum(len(s.notes) for s in segments),
                "clamped_durations_total": sum(s.clamped_durations for s in segments),
                "truncated_durations_total": sum(s.truncated_durations for s in segments),
                "max_delta_observed": max((s.max_delta for s in segments), default=0),
                "delta_limit": cfg.MAX_DELTA_UNITS,
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.slakh",
        description="Prepare a corpus from the Slakh stem subset.",
    )
    parser.add_argument("--source", type=Path, default=Path("new_dataset/slakh_subset"))
    parser.add_argument("--out", type=Path, default=Path("data/slakh"))
    parser.add_argument(
        "--grid", type=int, default=cfg.DEFAULT_GRID_CODE, choices=sorted(cfg.GRID_SUBDIVISIONS)
    )
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--keep-splits",
        type=Path,
        default=None,
        help="an earlier corpus.csv whose tracks keep their dev/test split; "
        "only tracks new to it are drawn",
    )
    args = parser.parse_args(argv)

    fixed = None
    if args.keep_splits is not None:
        # read before prepare(), which may be about to overwrite this very file
        with open(args.keep_splits, encoding="utf-8") as handle:
            fixed = {row["source_track"]: row["split"] for row in csv.DictReader(handle)}

    report = prepare(args.source, args.out, args.grid, args.dev_fraction, args.seed, fixed)
    segments = report.segments

    reasons: dict[str, int] = {}
    for rejection in report.rejections:
        reasons[rejection.reason] = reasons.get(rejection.reason, 0) + 1

    tracks = len({segment.source.track for segment in segments})
    lines = [
        f"scanned          {report.scanned} stems",
        f"rejected         {len(report.rejections)}"
        + ("  (" + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())) + ")" if reasons else ""),
        f"segmented        {report.segmented_tracks} stems cut at long rests",
    ]
    if report.tempo_map_stems:
        share = report.tempo_map_notes_kept / report.tempo_map_notes * 100.0
        lines.append(
            f"tempo changes    {report.tempo_map_stems} stems cut at tempo changes, "
            f"{report.tempo_map_notes_kept} of their {report.tempo_map_notes} notes kept ({share:.0f}%)"
        )
    lines.append(f"kept             {len(segments)} segments from {tracks} tracks")

    if segments:
        by_inst: dict[str, int] = {}
        by_grid: dict[int, int] = {}
        for segment in segments:
            by_inst[segment.source.inst_class] = by_inst.get(segment.source.inst_class, 0) + 1
            by_grid[segment.grid_code] = by_grid.get(segment.grid_code, 0) + 1
        notes = sorted(len(s.notes) for s in segments)
        spans = sorted(s.span_s for s in segments)
        lines += [
            "  instruments    " + ", ".join(f"{k} {v}" for k, v in sorted(by_inst.items())),
            "  grid chosen    "
            + ", ".join(f"1/{4 * cfg.GRID_SUBDIVISIONS[k]} x{v}" for k, v in sorted(by_grid.items())),
            f"  grid error     max {max(s.grid_error_ratio for s in segments):.2f} of a unit"
            f" (limit {MAX_GRID_ERROR_RATIO})",
            f"  notes          median {notes[len(notes) // 2]}, total {sum(notes)}",
            f"  span           median {spans[len(spans) // 2]:.0f} s, total {sum(spans) / 3600:.1f} h",
            f"  max delta      {max(s.max_delta for s in segments)} of {cfg.MAX_DELTA_UNITS} allowed",
            f"  rounded up     {sum(s.clamped_durations - s.truncated_durations for s in segments)} sub-unit notes to 1 grid unit",
            f"  truncated      {sum(s.truncated_durations for s in segments)} held notes at the 64-unit ceiling",
        ]

    lines.append(f"manifest         {Path(args.out) / 'corpus.csv'}")
    print("\n".join(lines))
    return 0 if segments else 1


if __name__ == "__main__":
    raise SystemExit(main())
