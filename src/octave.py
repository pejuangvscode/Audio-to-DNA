"""Strict and octave-tolerant accuracy over a finished experiment run (Subbab 3.3.5).

    python -m src.octave --data-dir data/slakh_v3 --split test --run results/slakh_test_v3

Strict mir_eval scoring counts a note one octave off as a total miss, and some
Slakh patches sound an octave away from the MIDI they were rendered from. This
separates those stems from ones the transcription actually failed on. It reads
only what the run left behind, the reference MIDI and the decoder's MIDI, so
nothing is transcribed or encoded again. The strict figure stays the headline;
this one is reported beside it, never in its place.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

from src import evaluate
from src.ablation import _format_p, _id, paired_p, per_track, track_of
from src.corpus import load_manifest
from src.reconstruct import read_midi

FIELDS = (
    "sample",
    "inst_class",
    "f_measure",
    "f_octave",
    "octave_shift",
    "f_measure_offset",
    "f_octave_offset",
)


def measure(reference_midi: Path, reconstructed_midi: Path) -> dict:
    reference = read_midi(reference_midi)
    estimated = read_midi(reconstructed_midi)
    strict = evaluate.accuracy(reference, estimated)
    shift, tolerant = evaluate.octave_tolerant(reference, estimated)
    return {
        "f_measure": round(strict.f_measure, 4),
        "f_octave": round(tolerant.f_measure, 4),
        "octave_shift": shift,
        "f_measure_offset": round(strict.f_measure_with_offset, 4),
        "f_octave_offset": round(tolerant.f_measure_with_offset, 4),
    }


def _p_text(value: float) -> str:
    text = _format_p(value)
    return text if text.startswith("<") else f"= {text}"


def summarise(rows: list[dict]) -> list[str]:
    """Markdown: overall, then one line per instrument class."""
    names = [row["sample"] for row in rows]
    strict = [row["f_measure"] for row in rows]
    tolerant = [row["f_octave"] for row in rows]
    shifted = [row for row in rows if row["octave_shift"] != 0]
    track_strict, track_tolerant = per_track(names, strict), per_track(names, tolerant)

    lines = [
        "# F-measure ketat dan toleran oktaf",
        "",
        f"{len(rows)} sampel dari {len(set(map(track_of, names)))} track. Toleran oktaf: "
        "seluruh not hasil rekonstruksi digeser bersama sejauh 0, ±12, atau ±24 semitone, "
        "dipilih yang F-nya tertinggi; seri dimenangkan pergeseran terkecil.",
        "",
        f"- F rerata per sampel: ketat {_id(statistics.fmean(strict), '.3f')}, "
        f"toleran oktaf {_id(statistics.fmean(tolerant), '.3f')}",
        f"- F rerata per track: ketat {_id(statistics.fmean(track_strict), '.3f')}, "
        f"toleran oktaf {_id(statistics.fmean(track_tolerant), '.3f')} "
        f"(Wilcoxon per track p {_p_text(paired_p(track_tolerant, track_strict))})",
        f"- Sampel yang paling cocok setelah digeser: {len(shifted)} dari {len(rows)}",
    ]
    for size, label in ((0.1, "0,1"), (0.3, "0,3")):
        count = sum(1 for row in shifted if row["f_octave"] - row["f_measure"] >= size)
        lines.append(f"  - dengan kenaikan F sedikitnya {label}: {count}")
    by_shift: dict[int, int] = {}
    for row in shifted:
        by_shift[row["octave_shift"]] = by_shift.get(row["octave_shift"], 0) + 1
    lines.append(
        "  - per pergeseran (semitone, positif berarti hasil transkripsi dinaikkan): "
        + ", ".join(f"{shift:+d}: {count}" for shift, count in sorted(by_shift.items()))
    )

    lines += [
        "",
        "| Kelas | Sampel | F ketat | F toleran oktaf | ΔF | Bergeser |",
        "|---|---|---|---|---|---|",
    ]
    classes: dict[str, list[dict]] = {}
    for row in rows:
        classes.setdefault(row["inst_class"], []).append(row)
    for name, members in sorted(classes.items(), key=lambda item: -len(item[1])):
        f_strict = statistics.fmean(row["f_measure"] for row in members)
        f_tolerant = statistics.fmean(row["f_octave"] for row in members)
        moved = sum(1 for row in members if row["octave_shift"] != 0)
        lines.append(
            f"| {name} | {len(members)} | {_id(f_strict, '.3f')} | {_id(f_tolerant, '.3f')} | "
            f"{_id(f_tolerant - f_strict, '+.3f')} | {moved} |"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.octave",
        description="Score a finished run again, allowing one octave shift per sample.",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--run", type=Path, required=True, help="output folder of src.experiment")
    parser.add_argument("--only", nargs="*", help="run only these samples, by name")
    args = parser.parse_args(argv)

    samples = [sample for sample in load_manifest(args.data_dir) if sample.split == args.split]
    if args.only:
        samples = [sample for sample in samples if sample.name in set(args.only)]

    rows: list[dict] = []
    missing: list[str] = []
    for index, sample in enumerate(samples, start=1):
        reconstructed = args.run / "reconstructed" / f"{sample.name}.mid"
        if not reconstructed.is_file() or not sample.midi_path.is_file():
            missing.append(sample.name)
            continue
        print(f"[{index}/{len(samples)}] {sample.name} ...", flush=True)
        row = {"sample": sample.name, "inst_class": sample.inst_class}
        row.update(measure(sample.midi_path, reconstructed))
        rows.append(row)

    if not rows:
        print("nothing was measured", file=sys.stderr)
        return 1

    with open(args.run / "octave.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    lines = summarise(rows)
    summary = args.run / "octave_summary.md"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # the unshifted score must be the run's own, or this is not the same measurement
    per_sample = args.run / "per_sample.csv"
    if per_sample.is_file():
        with open(per_sample, newline="", encoding="utf-8") as handle:
            recorded = {row["sample"]: float(row["f_measure"]) for row in csv.DictReader(handle) if row.get("f_measure")}
        agree = sum(1 for row in rows if abs(recorded.get(row["sample"], -1.0) - row["f_measure"]) < 1e-4)
        print(f"\nstrict f_measure vs experiment f_measure: {agree} of {len(rows)} samples agree")

    shifted = sum(1 for row in rows if row["octave_shift"] != 0)
    print(
        f"mean F strict {statistics.fmean(row['f_measure'] for row in rows):.4f}, "
        f"octave-tolerant {statistics.fmean(row['f_octave'] for row in rows):.4f}; "
        f"{shifted} of {len(rows)} samples fit better shifted"
    )
    # the tables carry Δ and ±, which a cp1252 console cannot print
    print(f"tables written to {summary}")
    if missing:
        print(f"{len(missing)} samples skipped, no reconstructed or reference MIDI: {', '.join(missing[:10])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
