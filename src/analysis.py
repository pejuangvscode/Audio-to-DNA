"""The statistical analysis of Tabel 3.14 over a finished test run (Subbab 3.3.5).

    python -m src.analysis --data-dir data/slakh_v3 --run results/slakh_test_v3 \\
        --characterization results/characterization_v3/per_sample.csv \\
        --output results/analysis_v3

Four questions, each answered on the track rather than the sample: the stems and
segments of one song share its tempo, its arrangement and its repeated sections,
so they are not independent observations.

    1. does quantisation change accuracy?            Wilcoxon signed-rank
    2. does repeat detection reduce size?            Wilcoxon signed-rank
    3. does performance differ between instruments?  Kruskal-Wallis, then
                                                     Mann-Whitney U per pair
    4. which data traits go with performance?        Spearman

Every p-value from all four is corrected together with Benjamini-Hochberg at
q = 0.05, as Subbab 3.3.5 specifies. Reads only finished outputs; nothing is
transcribed or encoded again.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import statistics
import sys
from pathlib import Path

from src.ablation import _id, benjamini_hochberg, paired_p, per_track, track_of
from src.characterize import NUMERIC
from src.corpus import load_manifest

Q = 0.05
# performance as Tabel 3.14 means it: accuracy, and the saving the token scheme
# adds over the transcribed MIDI (Persamaan 3.6); the total saving is above
# 99.4% on every sample and would rank nothing
OUTCOMES = (
    ("f_measure", "F-measure"),
    ("saving_token_scheme_pct", "S_token"),
)
LEVELS = ("rendah", "sedang", "tinggi")
# within-class correlations need enough songs to rank; smaller classes are
# left to the pooled analysis
MIN_CLASS_TRACKS = 30


def load(run: Path, data_dir: Path, split: str, characterization: Path | None) -> list[dict]:
    """per_sample.csv rows of the run, with instrument class and data traits added."""
    classes = {sample.name: sample.inst_class for sample in load_manifest(data_dir) if sample.split == split}
    traits: dict[str, dict] = {}
    if characterization is not None:
        with open(characterization, newline="", encoding="utf-8") as handle:
            traits = {row["sample"]: row for row in csv.DictReader(handle)}

    rows = []
    with open(run / "per_sample.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["sample"] not in classes:
                continue
            row["inst_class"] = classes[row["sample"]]
            for key in NUMERIC:
                if key in traits.get(row["sample"], {}):
                    row[f"trait:{key}"] = traits[row["sample"]][key]
            rows.append(row)
    return rows


def _values(rows: list[dict], column: str) -> list[float]:
    return [float(row[column]) if row.get(column, "") != "" else math.nan for row in rows]


def _track_means(rows: list[dict], column: str) -> list[float]:
    return per_track([row["sample"] for row in rows], _values(rows, column))


# --- questions 1 and 2: paired, on track means ---

def paired(rows: list[dict], after: str, before: str, label: str) -> dict:
    """`after` against `before` on the same tracks."""
    a, b = _track_means(rows, after), _track_means(rows, before)
    differences = [x - y for x, y in zip(a, b)]
    return {
        "question": label,
        "test": "Wilcoxon signed-rank",
        "units": len(a),
        "before": statistics.fmean(b),
        "after": statistics.fmean(a),
        "mean_difference": statistics.fmean(differences),
        "median_difference": statistics.median(differences),
        "up": sum(1 for d in differences if d > 0),
        "down": sum(1 for d in differences if d < 0),
        "p": paired_p(a, b),
    }


def by_repetition_level(rows: list[dict]) -> list[dict]:
    """Repeat detection per level of Tabel 3.4, descriptive.

    A sample whose payload shrinks without crossing a 223-byte codeword
    boundary saves tokens but no bases (Gambar 2.7); the share counts those.
    """
    out = []
    for level in LEVELS:
        members = [row for row in rows if row.get("pengulangan") == level]
        if not members:
            continue
        base_saving = _values(members, "base_saving_from_lz77_pct")
        out.append(
            {
                "level": level,
                "samples": len(members),
                "tracks": len({track_of(row["sample"]) for row in members}),
                "token_saving_pct": statistics.fmean(_values(members, "token_saving_pct")),
                "base_saving_pct": statistics.fmean(base_saving),
                "no_base_saving_share": sum(1 for value in base_saving if value == 0) / len(members),
            }
        )
    return out


# --- question 3: instruments ---

def class_units(rows: list[dict], column: str) -> dict[str, list[float]]:
    """One value per (track, class): a class's stems within a song are averaged,
    so a song with three guitar stems counts once for guitar."""
    groups: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        groups.setdefault((track_of(row["sample"]), row["inst_class"]), []).append(float(row[column]))
    units: dict[str, list[float]] = {}
    for (_, inst_class), values in sorted(groups.items()):
        units.setdefault(inst_class, []).append(statistics.fmean(values))
    return units


def instruments(rows: list[dict], column: str, label: str) -> tuple[dict, list[dict]]:
    from scipy.stats import kruskal, mannwhitneyu

    units = class_units(rows, column)
    n = sum(len(values) for values in units.values())
    k = len(units)
    statistic, p = kruskal(*units.values())
    overall = {
        "question": f"Perbedaan antarinstrumen: {label}",
        "test": "Kruskal-Wallis",
        "units": n,
        "groups": k,
        "H": float(statistic),
        # rank-based share of variance between classes, 0 to 1
        "epsilon_squared": (float(statistic) - k + 1) / (n - k),
        "p": float(p),
    }
    pairs = []
    for first, second in itertools.combinations(sorted(units), 2):
        a, b = units[first], units[second]
        u, p_pair = mannwhitneyu(a, b, alternative="two-sided")
        pairs.append(
            {
                "question": f"Pasangan instrumen: {label}",
                "test": "Mann-Whitney U",
                "outcome": label,
                "first": first,
                "second": second,
                "median_first": statistics.median(a),
                "median_second": statistics.median(b),
                # Persamaan 2.22 read as a share: the chance that a unit of the
                # first class beats one of the second
                "superiority": float(u) / (len(a) * len(b)),
                "p": float(p_pair),
            }
        )
    return overall, pairs


def class_table(rows: list[dict]) -> list[dict]:
    out = []
    per_outcome = {column: class_units(rows, column) for column, _ in OUTCOMES}
    for inst_class in sorted(per_outcome["f_measure"], key=lambda name: -len(per_outcome["f_measure"][name])):
        entry = {
            "inst_class": inst_class,
            "samples": sum(1 for row in rows if row["inst_class"] == inst_class),
            "tracks": len(per_outcome["f_measure"][inst_class]),
        }
        for column, _ in OUTCOMES:
            entry[f"median_{column}"] = statistics.median(per_outcome[column][inst_class])
        out.append(entry)
    return out


# --- question 4: data traits ---

def correlations(rows: list[dict], within: str | None = None) -> list[dict]:
    """Spearman between per-track means of each trait and of each outcome.

    Pooled over classes, a song's mean mixes its instruments, and the class
    alone explains much of the accuracy (question 3). Within one class every
    song appears once, so the units are independent and the class is held
    fixed; `within` names that class.
    """
    from scipy.stats import spearmanr

    if within is not None:
        rows = [row for row in rows if row["inst_class"] == within]
    scope = within or "semua kelas"
    out = []
    outcomes = {column: _track_means(rows, column) for column, _ in OUTCOMES}
    for trait in NUMERIC:
        key = f"trait:{trait}"
        if not any(key in row for row in rows):
            continue
        values = _track_means(rows, key)
        if any(math.isnan(value) for value in values) or len(set(values)) < 3:
            continue  # a trait that does not vary between tracks ranks nothing
        for column, label in OUTCOMES:
            if len(set(outcomes[column])) < 3:
                continue  # nor does an outcome that does not, and a NaN p would poison the correction
            rho, p = spearmanr(values, outcomes[column])
            out.append(
                {
                    "question": f"Korelasi ({scope}): {trait} dan {label}",
                    "test": "Spearman",
                    "scope": scope,
                    "trait": trait,
                    "outcome": label,
                    "units": len(values),
                    "rho": float(rho),
                    "p": float(p),
                }
            )
    return out


# --- report ---

def correct(tests: list[dict]) -> None:
    """Benjamini-Hochberg over every p-value at once, in place."""
    for test, q in zip(tests, benjamini_hochberg([test["p"] for test in tests])):
        test["q"] = q
        test["significant"] = q <= Q


def _q(value: float) -> str:
    return "< 0,001" if value < 0.001 else _id(value, ".3f")


def report(rows, quantisation, repetition, levels, kruskals, pairs, classes, spearman) -> list[str]:
    tests = [quantisation, *repetition, *kruskals, *pairs, *spearman]
    pooled = [test for test in spearman if test["scope"] == "semua kelas"]
    scopes = list(dict.fromkeys(test["scope"] for test in spearman if test["scope"] != "semua kelas"))
    tracks = len({track_of(row["sample"]) for row in rows})
    lines = [
        "# Analisis statistik (Tabel 3.14)",
        "",
        f"{len(rows)} sampel dari {tracks} track. Satuan analisis adalah track: stem dan potongan "
        "dari satu lagu tidak independen, sehingga dirata-ratakan per track sebelum diuji. "
        f"Seluruh {len(tests)} nilai p dikoreksi bersama dengan Benjamini–Hochberg; "
        f"q ≤ {_id(Q, '.2f')} dinyatakan signifikan.",
        "",
        "## 1. Apakah kuantisasi mengubah akurasi?",
        "",
        "| Track | F transkripsi | F rekonstruksi | ΔF rerata | ΔF median | Naik | Turun | q |",
        "|---|---|---|---|---|---|---|---|",
        f"| {quantisation['units']} | {_id(quantisation['before'], '.3f')} | {_id(quantisation['after'], '.3f')} | "
        f"{_id(quantisation['mean_difference'], '+.3f')} | {_id(quantisation['median_difference'], '+.3f')} | "
        f"{quantisation['up']} | {quantisation['down']} | {_q(quantisation['q'])} |",
        "",
        "## 2. Apakah deteksi pengulangan mengurangi ukuran?",
        "",
        "| Luaran | Track | Tanpa | Dengan | Selisih rerata | Turun | Naik | q |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for test in repetition:
        lines.append(
            f"| {test['question'].split(': ')[1]} | {test['units']} | {_id(test['before'], ',.0f')} | "
            f"{_id(test['after'], ',.0f')} | {_id(test['mean_difference'], '+,.0f')} | {test['down']} | "
            f"{test['up']} | {_q(test['q'])} |"
        )
    lines += [
        "",
        "Per taraf derajat pengulangan (Tabel 3.4), per sampel:",
        "",
        "| Taraf | Sampel | Track | Hemat token % | Hemat basa % | Tanpa hemat basa |",
        "|---|---|---|---|---|---|",
    ]
    for level in levels:
        lines.append(
            f"| {level['level']} | {level['samples']} | {level['tracks']} | {_id(level['token_saving_pct'], '.1f')} | "
            f"{_id(level['base_saving_pct'], '.1f')} | {_id(100 * level['no_base_saving_share'], '.1f')}% |"
        )

    lines += [
        "",
        "## 3. Apakah kinerja berbeda antarinstrumen?",
        "",
        "Satuan: satu nilai per pasangan (track, kelas). ε² adalah porsi variasi peringkat yang dijelaskan kelas.",
        "",
        "| Luaran | Satuan | Kelas | H | ε² | q |",
        "|---|---|---|---|---|---|",
    ]
    for test in kruskals:
        lines.append(
            f"| {test['question'].split(': ')[1]} | {test['units']} | {test['groups']} | {_id(test['H'], '.1f')} | "
            f"{_id(test['epsilon_squared'], '.3f')} | {_q(test['q'])} |"
        )
    lines += [
        "",
        "| Kelas | Sampel | Track | F median | S_token median (%) |",
        "|---|---|---|---|---|",
    ]
    for entry in classes:
        lines.append(
            f"| {entry['inst_class']} | {entry['samples']} | {entry['tracks']} | "
            f"{_id(entry['median_f_measure'], '.3f')} | {_id(entry['median_saving_token_scheme_pct'], '.1f')} |"
        )
    for _, label in OUTCOMES:
        mine = [pair for pair in pairs if pair["outcome"] == label]
        hits = sorted((pair for pair in mine if pair["significant"]), key=lambda pair: -abs(pair["superiority"] - 0.5))
        lines += [
            "",
            f"Pasangan kelas yang berbeda signifikan pada {label}: {len(hits)} dari {len(mine)}. "
            "Keunggulan adalah peluang satu satuan kelas pertama lebih tinggi daripada satu satuan kelas kedua.",
        ]
        if hits:
            lines += ["", "| Kelas pertama | Kelas kedua | Median | Keunggulan | q |", "|---|---|---|---|---|"]
            for pair in hits:
                spec = ".3f" if label == "F-measure" else ".1f"
                lines.append(
                    f"| {pair['first']} | {pair['second']} | {_id(pair['median_first'], spec)} vs "
                    f"{_id(pair['median_second'], spec)} | {_id(pair['superiority'], '.2f')} | {_q(pair['q'])} |"
                )

    lines += [
        "",
        "## 4. Karakteristik data apa yang berhubungan dengan kinerja?",
        "",
        "Korelasi Spearman antara rerata per track setiap karakteristik (Subbab 3.2.5) dan rerata per track luaran.",
        "",
        "| Karakteristik | ρ (F) | q (F) | ρ (S_token) | q (S_token) |",
        "|---|---|---|---|---|",
    ]
    by_trait: dict[str, dict[str, dict]] = {}
    for test in pooled:
        by_trait.setdefault(test["trait"], {})[test["outcome"]] = test
    def cells(test: dict | None) -> str:
        if test is None:
            return "– | –"
        return f"{_id(test['rho'], '+.2f')}{'*' if test['significant'] else ''} | {_q(test['q'])}"

    strongest = lambda item: -max(abs(test["rho"]) for test in item[1].values())  # noqa: E731
    for trait, found in sorted(by_trait.items(), key=strongest):
        lines.append(f"| {trait} | {cells(found.get('F-measure'))} | {cells(found.get('S_token'))} |")
    for _, label in OUTCOMES if scopes else ():
        lines += [
            "",
            f"Korelasi Spearman dengan {label} di dalam setiap kelas yang memuat sedikitnya "
            f"{MIN_CLASS_TRACKS} track; di dalam satu kelas setiap track muncul satu kali.",
            "",
            "| Karakteristik | " + " | ".join(scopes) + " |",
            "|---|" + "---|" * len(scopes),
        ]
        cells: dict[str, dict[str, dict]] = {}
        for test in spearman:
            if test["scope"] in scopes and test["outcome"] == label:
                cells.setdefault(test["trait"], {})[test["scope"]] = test
        for trait in sorted(cells, key=lambda name: -max(abs(test["rho"]) for test in cells[name].values())):
            lines.append(
                f"| {trait} | "
                + " | ".join(
                    f"{_id(cells[trait][scope]['rho'], '+.2f')}{'*' if cells[trait][scope]['significant'] else ''}"
                    if scope in cells[trait]
                    else "–"
                    for scope in scopes
                )
                + " |"
            )
    lines +=["", "\\* signifikan setelah koreksi."]
    return lines


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        fields += [key for key in row if key not in fields]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.analysis",
        description="Run the statistical tests of Tabel 3.14 on a finished run.",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--run", type=Path, required=True, help="output folder of src.experiment")
    parser.add_argument("--characterization", type=Path, default=None, help="per_sample.csv of src.characterize")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = load(args.run, args.data_dir, args.split, args.characterization)
    rows = [row for row in rows if row.get("f_measure", "") != ""]
    if not rows:
        print("no scored samples in the run for this split", file=sys.stderr)
        return 1

    quantisation = paired(rows, "f_measure", "f_transcription", "Kuantisasi dan akurasi")
    repetition = [
        paired(rows, "tokens_lz77", "tokens_plain", "Deteksi pengulangan: jumlah token"),
        paired(rows, "bases_pipeline", "bases_no_lz77", "Deteksi pengulangan: jumlah basa"),
    ]
    kruskals, pairs = [], []
    for column, label in OUTCOMES:
        overall, by_pair = instruments(rows, column, label)
        kruskals.append(overall)
        pairs += by_pair
    spearman = []
    if args.characterization:
        spearman = correlations(rows)
        tracks_per_class = {name: len(units) for name, units in class_units(rows, "f_measure").items()}
        for inst_class in sorted(tracks_per_class, key=lambda name: -tracks_per_class[name]):
            if tracks_per_class[inst_class] >= MIN_CLASS_TRACKS:
                spearman += correlations(rows, within=inst_class)
    correct([quantisation, *repetition, *kruskals, *pairs, *spearman])

    args.output.mkdir(parents=True, exist_ok=True)
    lines = report(rows, quantisation, repetition, by_repetition_level(rows), kruskals, pairs, class_table(rows), spearman)
    (args.output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(args.output / "tests.csv", [quantisation, *repetition, *kruskals, *spearman])
    _write_csv(args.output / "instrument_pairs.csv", pairs)

    significant = sum(1 for test in [quantisation, *repetition, *kruskals, *pairs, *spearman] if test["significant"])
    print(
        f"{len(rows)} samples, {len({track_of(row['sample']) for row in rows})} tracks; "
        f"{significant} of {1 + len(repetition) + len(kruskals) + len(pairs) + len(spearman)} tests significant at q = {Q}"
    )
    print(f"quantisation: mean F {quantisation['before']:.4f} -> {quantisation['after']:.4f}, q = {quantisation['q']:.3g}")
    for test in kruskals:
        print(f"{test['question']}: H = {test['H']:.1f}, q = {test['q']:.3g}")
    # the tables carry Δ and ρ, which a cp1252 console cannot print
    print(f"tables written to {args.output / 'analysis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
