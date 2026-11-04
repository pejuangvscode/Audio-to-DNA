"""The statistical analysis of Tabel 3.14, on the track as the unit."""

from __future__ import annotations

import pytest

from src import analysis


def row(sample, inst_class, f, saving=80.0, **extra):
    values = {
        "sample": sample,
        "inst_class": inst_class,
        "f_measure": str(f),
        "saving_token_scheme_pct": str(saving),
    }
    values.update({key: str(value) for key, value in extra.items()})
    return values


def test_a_track_with_many_stems_counts_once():
    rows = [row(f"Track001_S0{i}", "Piano", 0.5, f_transcription=0.4) for i in range(5)]
    rows += [row("Track002_S00", "Piano", 0.7, f_transcription=0.6)]
    result = analysis.paired(rows, "f_measure", "f_transcription", "test")
    assert result["units"] == 2
    assert result["up"] == 2
    assert result["mean_difference"] == pytest.approx(0.1)


def test_stems_of_one_class_in_one_song_are_averaged_into_one_unit():
    rows = [
        row("Track001_S00", "Guitar", 0.2),
        row("Track001_S01", "Guitar", 0.4),
        row("Track001_S02", "Piano", 0.9),
        row("Track002_S00", "Guitar", 0.6),
    ]
    units = analysis.class_units(rows, "f_measure")
    assert sorted(units["Guitar"]) == pytest.approx([0.3, 0.6])
    assert units["Piano"] == pytest.approx([0.9])


def test_clearly_separated_classes_differ_and_every_pair_is_tested():
    rows = []
    for track in range(12):
        rows.append(row(f"Track{track:03d}_S00", "Piano", 0.8 + track * 0.01))
        rows.append(row(f"Track{track:03d}_S01", "Bass", 0.4 + track * 0.01))
        rows.append(row(f"Track{track:03d}_S02", "Organ", 0.1 + track * 0.01))
    overall, pairs = analysis.instruments(rows, "f_measure", "F-measure")
    assert overall["groups"] == 3
    assert overall["p"] < 0.001
    assert len(pairs) == 3
    piano_over_bass = next(pair for pair in pairs if {pair["first"], pair["second"]} == {"Bass", "Piano"})
    # Bass sorts first, and every Bass unit is below every Piano unit
    assert piano_over_bass["superiority"] == pytest.approx(0.0)


def test_correlation_within_a_class_ignores_the_other_classes():
    rows = []
    for track in range(10):
        # within Piano the trait rises with F; Guitar runs the other way
        rows.append(row(f"Track{track:03d}_S00", "Piano", 0.1 * track, **{"trait:density": track}))
        rows.append(row(f"Track{track:03d}_S01", "Guitar", 1 - 0.1 * track, **{"trait:density": track}))
    within = analysis.correlations(rows, within="Piano")
    density_f = next(test for test in within if test["trait"] == "density" and test["outcome"] == "F-measure")
    assert density_f["rho"] == pytest.approx(1.0)
    assert density_f["units"] == 10
    assert density_f["scope"] == "Piano"


def test_a_trait_that_does_not_vary_is_skipped():
    rows = [row(f"Track{track:03d}_S00", "Piano", 0.1 * track, **{"trait:tempo": 120}) for track in range(6)]
    assert analysis.correlations(rows) == []


def test_every_p_value_is_corrected_together():
    tests = [{"p": 0.01}, {"p": 0.04}, {"p": 0.03}, {"p": 0.5}]
    analysis.correct(tests)
    assert all(test["q"] >= test["p"] for test in tests)
    # with m = 4: 0.01 * 4 / 1 = 0.04, and 0.03, 0.04 both become 0.04 * 4 / 3
    assert tests[0]["q"] == pytest.approx(0.04)
    assert tests[1]["q"] == pytest.approx(0.04 * 4 / 3)
    assert [test["significant"] for test in tests] == [True, False, False, False]
