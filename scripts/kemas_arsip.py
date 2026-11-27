"""Package the v3 results for a data archive such as Zenodo.

    python scripts/kemas_arsip.py

Copies the files described in DATA_DICTIONARY.md into arsip/audio-to-dna-data-v3/,
keeping their paths from the repository, writes a README and a SHA-256 list,
and zips the folder. Work files and the reconstructed MIDI (about 336 MB) are
left out, since decoding the archived FASTA files regenerates them.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "audio-to-dna-data-v3"
OUT = ROOT / "arsip"
FOLDER = OUT / NAME
ZIP = OUT / f"{NAME}.zip"

FILES = [
    "data/slakh_v3/corpus.csv",
    "data/slakh_v3/preflight.json",
    "results/calibration_v3/sweep_per_sample.csv",
    "results/calibration_v3/sweep_summary.csv",
    "results/slakh_test_v3/per_sample.csv",
    "results/slakh_test_v3/octave.csv",
    "results/slakh_test_v3/summary.md",
    "results/slakh_test_v3/octave_summary.md",
    "results/characterization_v3/per_sample.csv",
    "results/characterization_v3/summary_test.json",
    "results/analysis_v3/tests.csv",
    "results/analysis_v3/instrument_pairs.csv",
    "results/analysis_v3/analysis.md",
    "results/ablation_v3/ablation_per_sample.csv",
    "results/ablation_v3/ablation_summary.csv",
    "results/ablation_v3/ablation_summary.md",
    "DATA_DICTIONARY.md",
]
FASTA_DIR = "results/slakh_test_v3/fasta"


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip()


def test_samples() -> list[str]:
    with open(ROOT / "results/slakh_test_v3/per_sample.csv", newline="", encoding="utf-8") as handle:
        return [row["sample"] for row in csv.DictReader(handle)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def readme(commit: str, tag: str, dirty: bool, repo: str, samples: int) -> str:
    state = (
        "**Warning:** the working tree had uncommitted changes in the code when this archive was packed."
        if dirty
        else "The code had no uncommitted changes when this archive was packed."
    )
    return f"""# Audio-to-DNA: test results (data v3)

Results of an undergraduate thesis in Informatics at Universitas Pelita Harapan. The thesis builds a codec that stores music recordings in synthetic DNA by way of automatic music transcription: a recording is transcribed into notes with basic-pitch, quantised to a rhythmic grid, encoded as compact feature tokens with repeated phrases replaced by back-references, and mapped to oligonucleotides that satisfy the GC-content and homopolymer constraints, protected by a Reed–Solomon outer code and a rotating-code inner code.

The data come from every pitched stem of the 151 songs in the test partition of Slakh2100. After preparation they form {samples:,} test samples from 120 songs and 364 validation samples from 30 songs. This archive holds every per-sample measurement, the summaries and statistical tests behind the thesis's Chapter IV, and the DNA sequences the codec produced. No audio is included.

## Contents

The folder structure follows the code repository. Every file and column is described in `DATA_DICTIONARY.md`.

- `data/slakh_v3/`: the sample manifest and the data-preparation summary.
- `results/calibration_v3/`: onset-threshold calibration on the validation data.
- `results/slakh_test_v3/`: the main experiment on the test data, the octave-tolerant scores, and one FASTA file per test sample in `fasta/`.
- `results/characterization_v3/`: format, signal and musical-content traits of every sample.
- `results/analysis_v3/`: statistical tests, with the song as the unit.
- `results/ablation_v3/`: the ablation study.
- `SHA256SUMS.txt`: a SHA-256 digest of every file, to check that nothing changed.

## Provenance

- Code: {repo}
- Commit: `{commit}`{f" (tag `{tag}`)" if tag else ""}
- Packed: {date.today():%Y-%m-%d}

{state}

The repository README lists, in order, the commands that produce every file here. A FASTA file decodes back to MIDI and MusicXML with `python -m src.decode --input <sample>.fasta --output <sample>.mid`.

## Licence and citation

The data in this archive are released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0). They are derived from Slakh2100; please also cite it:

Manilow, E., Wichern, G., Seetharaman, P., and Le Roux, J. 2019. Cutting Music Source Separation Some Slakh: A Dataset to Study the Impact of Training Data Quality and Quantity. *2019 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)*, 45–49. https://doi.org/10.1109/WASPAA.2019.8937170
"""


def main() -> int:
    missing = [path for path in FILES if not (ROOT / path).is_file()]
    samples = test_samples()
    fasta = [f"{FASTA_DIR}/{name}.fasta{suffix}" for name in samples for suffix in ("", ".payload.json")]
    missing += [path for path in fasta if not (ROOT / path).is_file()]
    if missing:
        print("missing files, nothing packed:", *missing[:20], sep="\n  ")
        return 1

    dirty = bool(git("status", "--porcelain", "--", "src", "scripts", "tests", "requirements.txt"))
    commit, tag = git("rev-parse", "HEAD"), git("tag", "--points-at", "HEAD")
    repo = git("remote", "get-url", "origin").removesuffix(".git")
    if dirty:
        print("warning: the code has uncommitted changes; the archive README says so", file=sys.stderr)

    # FASTA files outside per_sample.csv belong to samples dropped after an
    # earlier run, so only the test samples' files are copied
    if FOLDER.exists():
        shutil.rmtree(FOLDER)
    for path in FILES + fasta:
        target = FOLDER / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
    (FOLDER / "README.md").write_text(readme(commit, tag, dirty, repo, len(samples)), encoding="utf-8")

    listed = sorted(p for p in FOLDER.rglob("*") if p.is_file())
    (FOLDER / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(p)}  {p.relative_to(FOLDER).as_posix()}\n" for p in listed), encoding="utf-8"
    )

    ZIP.unlink(missing_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(p for p in FOLDER.rglob("*") if p.is_file()):
            archive.write(path, f"{NAME}/{path.relative_to(FOLDER).as_posix()}")

    size = sum(p.stat().st_size for p in FOLDER.rglob("*") if p.is_file())
    print(f"{len(listed) + 1} files, {size / 1e6:.1f} MB, from commit {commit[:7]}{f' ({tag})' if tag else ''}")
    print(f"written {ZIP} ({ZIP.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
