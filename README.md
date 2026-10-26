# MP3-to-DNA

[![Tests](https://github.com/pejuangvscode/MP3-to-DNA/actions/workflows/tests.yml/badge.svg)](https://github.com/pejuangvscode/MP3-to-DNA/actions/workflows/tests.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)

A codec that stores music recordings in synthetic DNA by way of automatic music transcription. Instead of mapping the bytes of an audio file to nucleotides, it transcribes the recording into notes, encodes them as compact feature tokens with repeated phrases replaced by back-references, and maps the tokens to oligonucleotides that satisfy the GC-content and homopolymer constraints of DNA synthesis.

DNA storage is priced per synthesised nucleotide, and compressed audio is already close to its entropy, so the only way to write fewer bases is to change the representation. On 82 held-out stems from the Slakh dataset the codec needed 85.0% fewer bases than encoding the transcribed MIDI file, while the DNA layer itself was lossless.

This repository accompanies an undergraduate thesis in Informatics at Universitas Pelita Harapan.

![Codec architecture](Paper/fig1.png)

## How it works

| Stage | What it does | Lossless |
|---|---|---|
| Pre-processing | Decodes MP3 or FLAC, downmixes to mono, resamples to 22.05 kHz | no |
| Transcription | [basic-pitch](https://github.com/spotify/basic-pitch), onset threshold calibrated on a development split, 60 ms minimum note | no |
| Quantisation | Snaps notes to a 1/4, 1/8, 1/16 or 1/24 grid at a per-piece tempo; also writes MusicXML | no |
| Tokenisation | 22-bit note tokens with delta-coded onsets; LZ77 at token level replaces repeats with 17-bit references | yes |
| Outer code | Reed–Solomon RS(255, 223) over GF(2⁸) | yes |
| Inner code | 128-bit blocks to 81 trits, rotating code (no repeated base), SHA-256 scrambling with GC screening | yes |
| Output | 202-nt oligonucleotides (20-nt primers, 162-nt payload) written as FASTA | yes |

Decoding runs the lossless stages in reverse and writes MIDI and MusicXML. The design caps the effective density at 1.004 bits per base.

## Results

Test split: 82 piano and guitar stems from 71 Slakh tracks, 4.6 h of audio, 67,964 reference notes.

| Measure | Result |
|---|---|
| Token round trip | Bit-identical for all 82 samples |
| Constraint compliance | All 4,770 oligonucleotides within 40–60% GC; longest homopolymer 2 |
| Bases for the whole split | 963,540 |
| Saving against the transcribed MIDI file | 85.0% |
| Saving from repeat detection alone | 43.2 ± 25.2% per sample |
| Mean effective density | 0.839 bits per base (best sample 1.000) |
| Note-level F-measure (50 ms onset, 50 cents) | 0.748 |

Reconstruction error is dominated by transcription; quantisation has no significant effect on the whole-system F-measure. Two interactions limit the saving: Reed–Solomon codeword granularity hides repeat detection on short payloads, and transcription errors remove about a quarter of the repetition present in the reference notation.

## Installation

Developed and tested with Python 3.11. Transcription runs on the CPU, so no GPU is needed.

```bash
git clone https://github.com/pejuangvscode/MP3-to-DNA.git
cd MP3-to-DNA
pip install -r requirements.txt
```

## Usage

The tempo of the recording must be given, since tempo estimation is outside the scope of the codec.

```bash
python -m src.encode --input song.mp3 --tempo 120 --output song.fasta
python -m src.decode --input song.fasta --output song.mid
```

`--grid` selects the rhythmic grid (0 = 1/4, 1 = 1/8, 2 = 1/16, 3 = 1/24; default 2), and `--no-repeat-detection` turns off LZ77. Encoding also writes the transcription as MIDI and the quantised notes as MusicXML into a `work/` folder next to the FASTA file.

## Reproducing the experiments

The corpus manifest [`data/slakh_v2/corpus.csv`](data/slakh_v2/corpus.csv) lists every sample with its source track, tempo, grid, split and tercile labels. The audio is not in the repository; it is fetched from a Kaggle mirror of Slakh2100, which needs Kaggle credentials.

```bash
pip install kagglehub pyyaml
python scripts/fetch_slakh_subset.py --out new_dataset/slakh_subset --n 151
python -m src.slakh --source new_dataset/slakh_subset --out data/slakh_v2
python -m src.calibrate --data-dir data/slakh_v2 --split dev --output results/calibration_v2
python -m src.experiment --data-dir data/slakh_v2 --split test --onset-threshold 0.7 --output results/slakh_test_v2
python -m src.characterize --data-dir data/slakh_v2 --split test --output results/characterization_v2
python -m src.ablation --data-dir data/slakh_v2 --split test --work-dir results/slakh_test_v2/work --onset-threshold 0.7 --output results/ablation_v2 --transcribe
python Paper/figures.py
```

The steps are, in order: preparing the corpus (constant-tempo filter, per-piece tempo and grid fitting, cutting at long rests, development/test split), calibrating the onset threshold on the development split only, running the codec on the test split, characterising the corpus, running the ablation study, and drawing the paper figures.

## Repository layout

```
src/
  preprocess.py    audio loading, mono downmix, resampling
  transcribe.py    basic-pitch inference with an on-disk cache
  quantize.py      grid quantisation
  notation.py      MusicXML export
  tokenizer.py     token scheme, token-level LZ77, bit packing
  dna_codec.py     Reed-Solomon, scrambling, rotating code, FASTA
  encode.py        audio to FASTA
  decode.py        FASTA to MIDI and MusicXML
  reconstruct.py   quantised notes to MIDI
  evaluate.py      efficiency, compliance, accuracy, error decomposition
  baselines.py     comparison paths (audio bytes, transcribed MIDI, 2 bits per base)
  slakh.py         corpus preparation from Slakh
  corpus.py        manifest loading and validation
  characterize.py  quantitative corpus characterisation
  calibrate.py     onset-threshold calibration
  experiment.py    full evaluation run
  ablation.py      ablation study
  config.py        every design constant, in one place
tests/             unit tests
scripts/           dataset download
Paper/             conference paper source and figures
data/slakh_v2/     corpus manifest
```

## Tests

```bash
pytest -m "not slow"
pytest
```

The first command skips the tests that run the transcription model; the second runs everything.

## Citation

If you use this code, please cite it as described in [`CITATION.cff`](CITATION.cff), or use the "Cite this repository" button on GitHub.

## Acknowledgements

The evaluation uses [Slakh2100](http://www.slakh.com) (Manilow et al., WASPAA 2019), and transcription uses Spotify's [basic-pitch](https://github.com/spotify/basic-pitch) (Bittner et al., ICASSP 2022). Error correction relies on [reedsolo](https://github.com/tomerfiliba-org/reedsolomon), evaluation on [mir_eval](https://github.com/mir-evaluation/mir_eval), MusicXML export on [music21](https://github.com/cuthbertLab/music21), and FASTA output on [Biopython](https://biopython.org).

## License

See [LICENSE](LICENSE).
