"""Preparing the Slakh stem subset into an encodable corpus."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from src import config as cfg
from src.quantize import quantize
from src.slakh import (
    DURATION_LEVELS,
    MAGNITUDE_LEVELS,
    Report,
    Segment,
    StemSource,
    _assign_splits,
    _constant_tempo,
    _fit_grid,
    _segments_for,
    _silent_share,
    _split_points,
    _tempo_regions,
    _tercile_levels,
    scan,
)


class Note(NamedTuple):
    pitch: int
    start: float
    end: float


def on_grid(tempo: int, grid_code: int, positions: list[int], duration: int = 1):
    """Notes sitting exactly on the grid of `tempo` and `grid_code`."""
    delta = cfg.grid_unit_seconds(tempo, grid_code)
    return [
        Note(60, position * delta, (position + duration) * delta)
        for position in positions
    ]


def midi_with_tempi(tempi: list[float]) -> pretty_midi.PrettyMIDI:
    """A MIDI file carrying the given sequence of tempo events."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempi[0])
    for index, tempo in enumerate(tempi[1:], start=1):
        # pretty_midi exposes tempo changes only through the underlying ticks
        midi._tick_scales.append(
            (midi.time_to_tick(index * 1.0), 60.0 / (tempo * midi.resolution))
        )
    midi._update_tick_to_time(midi.time_to_tick(len(tempi) + 1))
    return midi


# --- tempo constancy ---

def test_a_single_tempo_is_constant():
    assert _constant_tempo(midi_with_tempi([120.0])) == 120


def test_a_tempo_repeated_is_still_constant():
    """Lakh MIDI restates the same tempo often; that is not a tempo map."""
    assert _constant_tempo(midi_with_tempi([120.0, 120.0, 120.0])) == 120


def test_a_changing_tempo_is_rejected():
    assert _constant_tempo(midi_with_tempi([120.0, 90.0])) is None


# --- cutting at tempo changes ---

def stem_with_tempo_change(tmp_path, change_tick: int, per_region: int = 40) -> StemSource:
    """A stem in sixteenths at 120 BPM, then at 90 BPM from `change_tick` on.

    The notes sit on the sixteenth grid counted in ticks from the start of the
    piece, as a sequencer writes them, whatever tick the change falls on.
    """
    midi = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=120.0)
    sixteenth = midi.resolution // 4
    midi._tick_scales.append((change_tick, 60.0 / (90.0 * midi.resolution)))
    first_after = -(-change_tick // sixteenth) * sixteenth
    ticks = [k * sixteenth for k in range(per_region)]
    ticks += [first_after + k * sixteenth for k in range(per_region)]
    midi._update_tick_to_time(ticks[-1] + 4 * midi.resolution)

    instrument = pretty_midi.Instrument(program=0)
    for tick in ticks:
        instrument.notes.append(
            pretty_midi.Note(100, 60, midi.tick_to_time(tick), midi.tick_to_time(tick + sixteenth))
        )
    midi.instruments.append(instrument)
    midi_path = tmp_path / "S00.mid"
    midi.write(str(midi_path))

    audio_path = tmp_path / "S00.flac"
    seconds = midi.tick_to_time(ticks[-1]) + 3.0
    sf.write(str(audio_path), np.zeros(int(seconds * 8000), dtype="float32"), 8000)
    return StemSource("Track00001", "S00", audio_path, midi_path, "Piano", "Piano", False)


def test_repeated_tempi_form_one_region():
    regions = _tempo_regions(midi_with_tempi([120.0, 120.0, 90.0]))
    assert [region.bpm for region in regions] == [120, 90]
    assert regions[0].origin_s == 0.0
    assert regions[0].end_s == regions[1].start_s == pytest.approx(2.0)


def test_a_tempo_map_is_cut_into_one_segment_per_tempo(tmp_path):
    stem = stem_with_tempo_change(tmp_path, change_tick=12 * 480)
    report = Report()
    segments = _segments_for(stem, cfg.DEFAULT_GRID_CODE, report)

    assert [(s.tempo, s.grid_code) for s in segments] == [(120, 2), (90, 2)]
    assert [len(s.notes) for s in segments] == [40, 40]
    # a file stores tempo in whole microseconds per beat, so 90 BPM drifts a little
    assert all(s.grid_error_ratio < 1e-3 for s in segments)
    assert report.tempo_map_notes_kept == report.tempo_map_notes == 80


def test_a_change_between_grid_lines_keeps_the_region_on_its_grid(tmp_path):
    """A change a thirty-second past a beat puts the new region's grid half a
    unit out of phase with the change. Measured from the change itself, the
    region would only fit at double tempo, 180 BPM.
    """
    stem = stem_with_tempo_change(tmp_path, change_tick=12 * 480 + 60)
    segments = _segments_for(stem, cfg.DEFAULT_GRID_CODE, Report())

    assert [(s.tempo, s.grid_code) for s in segments] == [(120, 2), (90, 2)]
    assert segments[1].grid_error_ratio < 1e-3


def test_a_tempo_map_with_no_long_region_is_rejected(tmp_path):
    stem = stem_with_tempo_change(tmp_path, change_tick=12 * 480, per_region=10)
    report = Report()
    assert _segments_for(stem, cfg.DEFAULT_GRID_CODE, report) == []
    assert report.rejections[-1].reason == "tempo_map"


# --- grid and tempo fitting ---

def test_a_piece_in_sixteenths_fits_the_sixteenth_grid():
    notes = on_grid(120, 2, [0, 1, 2, 3, 4, 6, 8, 9, 10, 12])
    tempo, grid_code, ratio = _fit_grid(notes, 120)
    assert (tempo, grid_code) == (120, 2)
    assert ratio == pytest.approx(0.0, abs=1e-9)


def test_a_piece_in_triplets_prefers_the_triplet_grid():
    """Six subdivisions to the beat, which four cannot express."""
    notes = on_grid(120, 3, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    _, grid_code, ratio = _fit_grid(notes, 120)
    assert grid_code == 3
    assert ratio == pytest.approx(0.0, abs=1e-9)


def test_a_piece_notated_at_half_tempo_is_refitted_to_double():
    """The offbeats of a piece written at half speed fall between grid lines."""
    notes = on_grid(140, 2, [0, 1, 2, 3, 5, 7, 9, 11, 13, 15])
    tempo, _, ratio = _fit_grid(notes, 70)
    assert tempo == 140
    assert ratio == pytest.approx(0.0, abs=1e-9)


def test_fitting_never_leaves_the_tempo_field_range():
    notes = on_grid(280, 2, [0, 1, 2, 3, 4])
    tempo, _, _ = _fit_grid(notes, 280)
    assert cfg.MIN_TEMPO <= tempo <= cfg.MAX_TEMPO


# --- cutting at long rests ---

def test_a_rest_within_the_delta_field_is_not_cut():
    notes = on_grid(120, 2, [0, 1, cfg.MAX_DELTA_UNITS])
    assert _split_points(notes, 120, 2) == []


def test_a_rest_beyond_the_delta_field_is_cut():
    notes = on_grid(120, 2, [0, 1, 1 + cfg.MAX_DELTA_UNITS + 1])
    assert _split_points(notes, 120, 2) == [2]


def test_a_late_first_entry_is_cut():
    """The first delta is measured from zero, so a late entry overflows too."""
    notes = on_grid(120, 2, [cfg.MAX_DELTA_UNITS + 1, cfg.MAX_DELTA_UNITS + 2])
    assert _split_points(notes, 120, 2) == [0]


def test_cutting_on_a_grid_unit_preserves_the_fit():
    """Shifting a segment to its own timeline must not move it off the grid.

    A cut at an arbitrary second subtracts a fractional number of grid units
    from every onset, which puts the whole segment out of phase and undoes the
    tempo and grid fit.
    """
    tempo, grid_code = 120, 2
    unit = cfg.grid_unit_seconds(tempo, grid_code)
    notes = on_grid(tempo, grid_code, [40, 41, 42, 44, 46])

    aligned = int((notes[0].start - 1.0) / unit) * unit
    shifted = [Note(n.pitch, n.start - aligned, n.end - aligned) for n in notes]
    assert quantize(shifted, tempo, grid_code).rms_grid_error_ms == pytest.approx(0.0, abs=1e-6)

    off_grid = notes[0].start - 1.0 + unit / 2.0
    nudged = [Note(n.pitch, n.start - off_grid, n.end - off_grid) for n in notes]
    assert quantize(nudged, tempo, grid_code).rms_grid_error_ms > 1.0


# --- audibility ---

def test_notes_written_where_the_audio_is_silent_are_counted(tmp_path):
    """A tone for the first second, then nothing, as a patch out of range."""
    rate = 8000
    signal = np.zeros(3 * rate, dtype="float32")
    signal[:rate] = 0.3 * np.sin(2 * np.pi * 220.0 * np.arange(rate) / rate)
    audio_path = tmp_path / "S00.flac"
    sf.write(str(audio_path), signal, rate)

    source = StemSource("Track00001", "S00", audio_path, None, "Brass", "Trombone", False)
    notes = [pretty_midi.Note(100, 60, start, start + 0.3) for start in (0.1, 0.5, 1.5, 2.2)]
    whole = Segment(source, -1, 0.0, 3.0, 120, cfg.DEFAULT_GRID_CODE, notes)
    assert _silent_share(whole) == 0.5

    # a cut segment is read from its own slice of the stem
    tail = Segment(source, 0, 1.0, 3.0, 120, cfg.DEFAULT_GRID_CODE, notes[2:])
    assert _silent_share(tail) == 1.0


# --- stratification ---

def test_terciles_populate_every_level():
    levels = _tercile_levels([float(value) for value in range(30)], MAGNITUDE_LEVELS)
    assert set(levels) == set(MAGNITUDE_LEVELS)
    assert len(levels) == 30


def test_terciles_order_low_to_high():
    levels = _tercile_levels([9.0, 1.0, 5.0], DURATION_LEVELS)
    assert levels == [DURATION_LEVELS[2], DURATION_LEVELS[0], DURATION_LEVELS[1]]


def test_terciles_of_nothing_is_nothing():
    assert _tercile_levels([], MAGNITUDE_LEVELS) == []


# --- several stems per track ---

def make_track(root, name, stems):
    """A Slakh-style track directory; `stems` maps stem id to instrument class.
    Only file presence and metadata matter to scan(), so the files are empty.
    """
    track = root / name
    (track / "stems").mkdir(parents=True)
    (track / "MIDI").mkdir()
    lines = ["stems:"]
    for stem_id, inst_class in stems.items():
        (track / "stems" / f"{stem_id}.flac").write_bytes(b"")
        (track / "MIDI" / f"{stem_id}.mid").write_bytes(b"")
        lines += [f"  {stem_id}:", f"    inst_class: {inst_class}", "    is_drum: false"]
    (track / "metadata.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def segment(track, stem_id, inst_class):
    source = StemSource(track, stem_id, None, None, inst_class, inst_class, False)
    return Segment(source, -1, 0.0, 1.0, 120, cfg.DEFAULT_GRID_CODE, [])


def test_scan_reads_every_stem_of_a_track(tmp_path):
    make_track(tmp_path, "Track00001", {"S00": "Bass", "S01": "Piano", "S02": "Guitar"})
    stems = scan(tmp_path)
    assert [(s.stem_id, s.inst_class) for s in stems] == [
        ("S00", "Bass"), ("S01", "Piano"), ("S02", "Guitar")
    ]


def test_scan_skips_a_stem_without_midi(tmp_path):
    make_track(tmp_path, "Track00001", {"S00": "Bass", "S01": "Piano"})
    (tmp_path / "Track00001" / "MIDI" / "S00.mid").unlink()
    assert [s.stem_id for s in scan(tmp_path)] == ["S01"]


def test_stems_of_one_track_never_straddle_the_split():
    segments = [
        segment(f"Track{t:05d}", stem_id, inst_class)
        for t in range(10)
        for stem_id, inst_class in (("S00", "Bass"), ("S01", "Piano"), ("S02", "Guitar"))
    ]
    splits = _assign_splits(segments, dev_fraction=0.2, seed=0)

    by_track = {}
    for seg, split in zip(segments, splits):
        by_track.setdefault(seg.source.track, set()).add(split)
    assert all(len(found) == 1 for found in by_track.values())
    assert sum(found == {"dev"} for found in by_track.values()) == 2


def test_kept_splits_stay_and_only_new_tracks_are_drawn():
    segments = [
        segment(f"Track{t:05d}", stem_id, inst_class)
        for t in range(20)
        for stem_id, inst_class in (("S00", "Bass"), ("S01", "Piano"))
    ]
    fixed = {f"Track{t:05d}": "dev" if t < 2 else "test" for t in range(10)}
    splits = _assign_splits(segments, dev_fraction=0.2, seed=0, fixed=fixed)

    by_track = {}
    for seg, split in zip(segments, splits):
        by_track.setdefault(seg.source.track, set()).add(split)
    assert all(by_track[track] == {split} for track, split in fixed.items())
    new_dev = [t for t in by_track if t not in fixed and by_track[t] == {"dev"}]
    assert len(new_dev) == 2


def test_one_stem_per_track_is_still_stratified_by_instrument():
    segments = [segment(f"Track{t:05d}", "S00", "Piano") for t in range(5)]
    segments += [segment(f"Track{t:05d}", "S00", "Guitar") for t in range(5, 10)]
    splits = _assign_splits(segments, dev_fraction=0.2, seed=0)

    dev_classes = [seg.source.inst_class for seg, split in zip(segments, splits) if split == "dev"]
    assert sorted(dev_classes) == ["Guitar", "Piano"]
