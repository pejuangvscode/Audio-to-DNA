"""Package the v3 results for a data archive such as Zenodo.

    python scripts/kemas_arsip.py

Writes the files to upload into arsip/zenodo/: README.md and DATA_DICTIONARY.md
on their own, so the archive page shows them; one zip of the tables and
summaries; one zip of the FASTA files; and SHA256SUMS.txt for every file in
both zips. Zenodo previews a zip only below a file count that the 2,422 FASTA
files exceed, hence the split. Both zips unpack into the same folder,
audio-to-dna-data-v3/, laid out as in the repository.

Work files and the reconstructed MIDI (about 336 MB) are left out, since
decoding the archived FASTA files regenerates them.
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
UPLOAD = OUT / "zenodo"
ZIP_DATA = UPLOAD / f"{NAME}.zip"
ZIP_FASTA = UPLOAD / "audio-to-dna-fasta-v3.zip"

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
# the code whose output this is; packing tools may change after it, the codec may not
CODE = ["src", "requirements.txt"]


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


def provenance() -> tuple[str, list[str]]:
    """Markdown lines naming the code, plus warnings for the console."""
    commit = git("rev-parse", "HEAD")
    tag = git("describe", "--tags", "--abbrev=0")
    repo = git("remote", "get-url", "origin").removesuffix(".git")
    lines = [f"- Code: {repo}"]
    warnings = []
    if tag:
        tagged = git("rev-list", "-n", "1", tag)
        lines.append(f"- Results produced with: tag `{tag}` (commit `{tagged[:7]}`)")
        if git("diff", "--name-only", tag, "HEAD", "--", *CODE):
            warnings.append(f"the codec changed after {tag}")
            lines.append(f"- **Warning:** `src/` or `requirements.txt` changed between `{tag}` and the packing commit.")
    else:
        warnings.append("no tag found; the archive README names the packing commit only")
    lines.append(f"- Packed from commit: `{commit}`, on {date.today():%Y-%m-%d}")
    if git("status", "--porcelain", "--", *CODE):
        warnings.append("the codec has uncommitted changes")
        lines.append("- **Warning:** the codec had uncommitted changes when this archive was packed.")
    return "\n".join(lines), warnings


def readme(provenance_md: str, samples: int) -> str:
    return f"""# Audio-to-DNA: test results (data v3)

Results of an undergraduate thesis in Informatics at Universitas Pelita Harapan. The thesis builds a codec that stores music recordings in synthetic DNA by way of automatic music transcription: a recording is transcribed into notes with basic-pitch, quantised to a rhythmic grid, encoded as compact feature tokens with repeated phrases replaced by back-references, and mapped to oligonucleotides that satisfy the GC-content and homopolymer constraints, protected by a Reed–Solomon outer code and a rotating-code inner code.

The data come from every pitched stem of the 151 songs in the test partition of Slakh2100. After preparation they form {samples:,} test samples from 120 songs and 364 validation samples from 30 songs. This archive holds every per-sample measurement, the summaries and statistical tests behind the thesis's Chapter IV, and the DNA sequences the codec produced. No audio is included.

## Files

| File | Contents |
|---|---|
| `README.md` | This description |
| `DATA_DICTIONARY.md` | Every file and column, with units |
| `{ZIP_DATA.name}` | The tables and summaries, plus both documents above |
| `{ZIP_FASTA.name}` | One FASTA file per test sample, with the digest of its token payload |
| `SHA256SUMS.txt` | A SHA-256 digest of every file inside both zips |

Both zips unpack into the same folder, `{NAME}/`, laid out as in the code repository:

- `data/slakh_v3/`: the sample manifest and the data-preparation summary.
- `results/calibration_v3/`: onset-threshold calibration on the validation data.
- `results/slakh_test_v3/`: the main experiment on the test data and the octave-tolerant scores; `fasta/` holds the DNA sequences.
- `results/characterization_v3/`: format, signal and musical-content traits of every sample.
- `results/analysis_v3/`: statistical tests, with the song as the unit.
- `results/ablation_v3/`: the ablation study.

To check the files after unpacking both zips, run `sha256sum -c SHA256SUMS.txt` inside `{NAME}/`.

## Provenance

{provenance_md}

The repository README lists, in order, the commands that produce every file here. A FASTA file decodes back to MIDI and MusicXML with `python -m src.decode --input <sample>.fasta --output <sample>.mid`.

## Licence and citation

The data in this archive are released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0). They are derived from Slakh2100; please also cite it:

Manilow, E., Wichern, G., Seetharaman, P., and Le Roux, J. 2019. Cutting Music Source Separation Some Slakh: A Dataset to Study the Impact of Training Data Quality and Quantity. *2019 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)*, 45–49. https://doi.org/10.1109/WASPAA.2019.8937170
"""


def write_zip(target: Path, members: list[Path]) -> None:
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in members:
            archive.write(path, f"{NAME}/{path.relative_to(FOLDER).as_posix()}")


def main() -> int:
    missing = [path for path in FILES if not (ROOT / path).is_file()]
    samples = test_samples()
    fasta = [f"{FASTA_DIR}/{name}.fasta{suffix}" for name in samples for suffix in ("", ".payload.json")]
    missing += [path for path in fasta if not (ROOT / path).is_file()]
    if missing:
        print("missing files, nothing packed:", *missing[:20], sep="\n  ")
        return 1

    provenance_md, warnings = provenance()
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    # FASTA files outside per_sample.csv belong to samples dropped after an
    # earlier run, so only the test samples' files are copied
    for old in (FOLDER, UPLOAD):
        if old.exists():
            shutil.rmtree(old)
    for path in FILES + fasta:
        target = FOLDER / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
    (FOLDER / "README.md").write_text(readme(provenance_md, len(samples)), encoding="utf-8")

    every = sorted(p for p in FOLDER.rglob("*") if p.is_file())
    sums = "".join(f"{sha256(p)}  {p.relative_to(FOLDER).as_posix()}\n" for p in every)
    (FOLDER / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")

    UPLOAD.mkdir(parents=True)
    in_fasta = [p for p in every if p.relative_to(FOLDER).as_posix().startswith(FASTA_DIR + "/")]
    in_data = [p for p in every if p not in set(in_fasta)] + [FOLDER / "SHA256SUMS.txt"]
    write_zip(ZIP_DATA, in_data)
    write_zip(ZIP_FASTA, in_fasta)
    for name in ("README.md", "DATA_DICTIONARY.md", "SHA256SUMS.txt"):
        shutil.copy2(FOLDER / name, UPLOAD / name)

    print(f"upload these {len(list(UPLOAD.iterdir()))} files from {UPLOAD}:")
    for path in sorted(UPLOAD.iterdir()):
        count = f", {len(zipfile.ZipFile(path).namelist())} files inside" if path.suffix == ".zip" else ""
        print(f"  {path.name} ({path.stat().st_size / 1e6:.2f} MB{count})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
