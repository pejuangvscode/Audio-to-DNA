# Data dictionary

This document describes every file in the results archive of the Audio-to-DNA codec, as used in the undergraduate thesis at Universitas Pelita Harapan and the accompanying ICACSIS paper. It lists what one row of each file stands for and what every column means, with its unit. Equation, table and section numbers (for example Eq. 3.4) refer to the thesis.

## Conventions

| Item | Convention |
|---|---|
| Decimal mark | Point (`0.554`) |
| Percentages | 0–100, in columns whose name ends in `_pct` |
| Proportions | 0–1 (GC content, F-measure, precision, recall, shares) |
| Booleans | `True` or `False` |
| Empty cell | Not applicable to that row |
| Sample name | `TrackNNNNN_SXX` is one Slakh stem; a suffix `_sNN` marks a segment of that stem, cut at a tempo change or at a rest longer than 255 grid units |
| Song | The Slakh track (`TrackNNNNN`), the unit of every statistical test |
| Split | `dev` = validation data, used only to calibrate the onset threshold; `test` = test data, on which every reported result is measured |
| Instrument class | Slakh's own class names, e.g. `Strings (continued)` for string ensembles, choirs and synth strings |
| Tercile levels | Duration: `pendek`/`sedang`/`panjang` (short/medium/long). Note density and repetition: `rendah`/`sedang`/`tinggi` (low/medium/high). Boundaries are in `summary_test.json` under `level_boundaries` |

The audio is never redistributed. Every audio-derived figure was measured on Slakh2100 stems (Manilow et al. 2019), fetched with `scripts/fetch_slakh_subset.py`.

## Contents

| File | One row is | Rows |
|---|---|---|
| `data/slakh_v3/corpus.csv` | One sample (validation and test) | 1,575 |
| `data/slakh_v3/preflight.json` | Summary of data preparation, plus every rejected stem or segment | – |
| `results/calibration_v3/sweep_per_sample.csv` | One validation sample at one onset threshold | 2,184 |
| `results/calibration_v3/sweep_summary.csv` | One onset threshold | 6 |
| `results/slakh_test_v3/per_sample.csv` | One test sample, every measurement of the main experiment | 1,211 |
| `results/slakh_test_v3/octave.csv` | One test sample, strict and octave-tolerant F-measure | 1,211 |
| `results/slakh_test_v3/fasta/*.fasta` | One test sample's DNA sequences, one record per oligo | 1,211 files |
| `results/slakh_test_v3/fasta/*.fasta.payload.json` | Length and digest of the token payload stored in that FASTA | 1,211 files |
| `results/characterization_v3/per_sample.csv` | One sample, its format, signal and musical-content traits | 1,575 |
| `results/characterization_v3/summary_test.json` | Per-class statistics of the test data, and the tercile boundaries | – |
| `results/analysis_v3/tests.csv` | One statistical test (Wilcoxon, Kruskal–Wallis or Spearman) | 305 |
| `results/analysis_v3/instrument_pairs.csv` | One Mann–Whitney comparison between two instrument classes | 132 |
| `results/ablation_v3/ablation_per_sample.csv` | One test sample under one ablation variant (17 variants × 1,211 samples, plus 633 for `notated_tempo`) | 21,220 |
| `results/ablation_v3/ablation_summary.csv` | One ablation variant | 18 |
| `*.md` next to the CSV files | Human-readable reports written by the same programs | – |

## `data/slakh_v3/corpus.csv`

The manifest written by `src/slakh.py`. Every later program reads the sample list from here.

| Column | Meaning | Unit |
|---|---|---|
| `sample` | Sample name | – |
| `tempo` | Tempo used for quantisation and stored in the token header, chosen per sample from the written tempo, half of it or double it | BPM |
| `grid` | Grid resolution code: 0 = 1/4, 1 = 1/8, 2 = 1/16, 3 = 1/24 note | – |
| `durasi` | Tercile level of the notation span | – |
| `kerapatan` | Tercile level of the note density | – |
| `pengulangan` | Tercile level of the repetition degree of the reference notation | – |
| `split` | `dev` or `test` | – |
| `inst_class` | Slakh instrument class | – |
| `source_track` | Slakh track (song) | – |
| `segment` | `whole` for an uncut stem, otherwise the segment index | – |
| `notes` | Notes in the reference MIDI of the sample | count |
| `span_s` | First onset to last note-off of the reference notation | s |
| `audio_path`, `midi_path` | Location of the stem audio and its reference MIDI in the local Slakh copy | – |

## `data/slakh_v3/preflight.json`

| Key | Meaning |
|---|---|
| `source` | Folder of the local Slakh copy |
| `grid_code` | Default grid code before per-sample fitting |
| `seed` | Seed of the song-level split |
| `tracks_with_kept_split` | Songs whose split was carried over from the previous manifest |
| `stems_scanned` | Pitched stems examined |
| `stems_with_tempo_map` | Stems whose MIDI changes tempo, cut into constant-tempo regions |
| `stems_segmented` | Stems cut at rests longer than 255 grid units |
| `tempo_map_notes`, `tempo_map_notes_kept` | Notes in tempo-changing stems, and how many survived in kept regions |
| `segments_kept`, `tracks_kept` | Samples and songs in the final manifest |
| `dev`, `test` | Samples per split |
| `notes_total` | Reference notes over all kept samples |
| `clamped_durations_total`, `truncated_durations_total` | Reference durations clamped to 1–64 grid units, and those cut at 64 |
| `max_delta_observed`, `delta_limit` | Largest gap between consecutive onsets, and the 255-unit limit of the 8-bit field |
| `rejections` | One entry per rejected stem or segment: `track` (name), `reason`, `detail`. Reasons: `tempo_map` (no tempo region with at least 32 notes), `off_grid` (onsets deviate more than 0.25 grid units after fitting), `too_short` (no segment with at least 32 notes), `inaudible` (more than 10% of reference notes start below −60 dBFS) |

## `results/calibration_v3/`

`sweep_per_sample.csv`: the 364 validation samples at each onset threshold from 0.30 to 0.80. Scores compare the quantised notes with the reference notation.

| Column | Meaning | Unit |
|---|---|---|
| `threshold` | basic-pitch onset threshold | 0–1 |
| `sample`, `inst_class` | Sample name and instrument class | – |
| `reference_notes` | Notes in the reference MIDI | count |
| `transcribed_notes` | Notes output by basic-pitch | count |
| `quantized_notes` | Notes after quantisation | count |
| `precision`, `recall`, `f_measure` | Note-level scores, onset within ±50 ms and pitch within ±50 cents (mir_eval defaults) | 0–1 |
| `f_measure_offset` | F-measure that also requires the offset within max(20% of the reference duration, 50 ms) | 0–1 |

`sweep_summary.csv`: one row per threshold. `samples` (count), `mean_f`, `sd_f` (sample standard deviation), `median_f`, `mean_precision`, `mean_recall`, `mean_f_offset`, and `mean_notes_ratio` (mean over samples of transcribed notes divided by reference notes). The threshold with the highest `mean_f`, 0.70, was used on the test data.

## `results/slakh_test_v3/per_sample.csv`

Written by `src/experiment.py`, one row per test sample.

| Column | Meaning | Unit |
|---|---|---|
| `sample` | Sample name | – |
| `tempo` | Tempo stored in the token header | BPM |
| `durasi`, `kerapatan`, `pengulangan` | Tercile levels, as in `corpus.csv` | – |
| `audio_duration_s` | Duration of the decoded audio | s |
| `leading_silence_s` | Time before the first sample louder than −60 dBFS | s |
| `mp3_bytes` | Size of the input audio file. The name is historical: every input is FLAC | bytes |
| `transcribed_notes` | Notes output by basic-pitch | count |
| `offset_ms` | Circular mean of the transcribed onsets' phase relative to the grid; positive means late. Diagnostic only, never applied | ms |
| `offset_concentration` | Resultant length of those phases: 0 = no common shift, 1 = every onset shifted alike | 0–1 |
| `quantized_notes` | Notes after quantisation | count |
| `merged_notes` | Notes merged into another of the same pitch and grid position | count |
| `clamped_durations` | Durations clamped to 1–64 grid units | count |
| `rms_grid_error_ms` | Root-mean-square distance between the transcribed onsets and their grid lines | ms |
| `grid_unit_ms` | Length of one grid unit Δ (Eq. 3.1) | ms |
| `tokens_lz77` | Tokens after repeat detection, note and reference tokens together | count |
| `note_tokens`, `reference_tokens` | The two kinds of token in `tokens_lz77` | count |
| `payload_bytes` | Packed header and tokens | bytes |
| `oligos` | Oligos in the FASTA file | count |
| `bases_pipeline` | Bases to synthesise, `oligos` × 202 | bases |
| `rescrambles` | Oligos that had to be scrambled again at GC screening | count |
| `encode_s` | Encoding time, transcription included | s |
| `tokens_plain`, `payload_bytes_plain` | Tokens and payload with repeat detection switched off | count, bytes |
| `token_saving_pct` | (1 − `tokens_lz77` / `tokens_plain`) × 100 | % |
| `bases_no_lz77` | Bases with repeat detection switched off | bases |
| `base_saving_from_lz77_pct` | (1 − `bases_pipeline` / `bases_no_lz77`) × 100 | % |
| `lossless` | The payload decoded from the FASTA matches the length and SHA-256 digest of the payload that was encoded | – |
| `deterministic` | Encoding the same tokens a second time gives identical oligos | – |
| `decode_s` | Decoding time | s |
| `reconstructed_notes` | Notes recovered by decoding | count |
| `gc_mean`, `gc_min`, `gc_max` | GC content per oligo, primers included: mean, minimum and maximum over the sample's oligos | 0–1 |
| `gc_in_range` | Share of oligos with GC content within 0.40–0.60 | 0–1 |
| `max_homopolymer` | Longest run of one base in any oligo | bases |
| `rejection_rate` | `rescrambles` / `oligos` | 0–1 |
| `compliant` | Every oligo meets both the GC range and a homopolymer of at most 3 | – |
| `bases_b1_direct` | Comparison B1: bases if the audio file's bytes went through the same codec, computed with Eq. 3.2 | bases |
| `bases_b2_symbolic` | Comparison B2: bases for the transcribed MIDI file, before quantisation, through the same codec | bases |
| `bases_b3_theoretical` | Comparison B3: 2 bits per base without constraints or error correction, i.e. audio bytes × 4 | bases |
| `b1_addressable` | Whether B1 fits within the 65,536 oligos a 2-byte index can address | – |
| `saving_total_pct` | (1 − `bases_pipeline` / `bases_b1_direct`) × 100, Eq. 3.4 | % |
| `saving_transcription_pct` | (1 − `bases_b2_symbolic` / `bases_b1_direct`) × 100, Eq. 3.5 | % |
| `saving_token_scheme_pct` | (1 − `bases_pipeline` / `bases_b2_symbolic`) × 100, Eq. 3.6 | % |
| `effective_density` | `payload_bytes` × 8 / `bases_pipeline`, Eq. 2.2 | bits/base |
| `reference_notes` | Notes in the reference MIDI | count |
| `precision`, `recall`, `f_measure` | Reconstructed notes against the reference, onset within ±50 ms and pitch within ±50 cents | 0–1 |
| `precision_offset`, `recall_offset`, `f_measure_offset` | The same, also requiring the offset within max(20% of the reference duration, 50 ms) | 0–1 |
| `f_transcription` | F-measure of the raw transcription against the reference, before quantisation | 0–1 |
| `f_quantization` | F-measure of the quantised notes against the raw transcription | 0–1 |
| `total_s` | Time for the whole sample, comparisons and checks included | s |

## `results/slakh_test_v3/octave.csv`

Written by `src/octave.py`. The octave-tolerant score moves the whole transcription by one shift of 0, ±12 or ±24 semitones and keeps the best F-measure; a tie goes to the smaller shift. A single note in the wrong octave still counts as wrong.

| Column | Meaning | Unit |
|---|---|---|
| `sample`, `inst_class` | Sample name and instrument class | – |
| `f_measure` | Strict F-measure, as in `per_sample.csv` | 0–1 |
| `f_octave` | Octave-tolerant F-measure | 0–1 |
| `octave_shift` | Shift applied to the transcription; +12 means the transcription sat one octave below the reference | semitones |
| `f_measure_offset`, `f_octave_offset` | The same pair with the offset criterion | 0–1 |

## `results/slakh_test_v3/fasta/`

`<sample>.fasta` holds the system's output: one record per oligo, named `oligo_NNNNN`. Each oligo is 202 nt: the forward primer `ACGTAGCTAGCATGCATCGA` (20 nt), a 162-nt payload, and the reverse primer `TGCATCGATCGTACGATGCA` (20 nt). The payload carries a 2-byte index, a 1-byte seed counter and 29 bytes of Reed–Solomon-coded data, mapped to bases with a rotating code.

`<sample>.fasta.payload.json` stores `payload_bytes` and `sha256` of the token payload that was encoded, so a decoded file can be checked without the encoder. To decode one file with the code repository:

```bash
python -m src.decode --input Track01876_S00.fasta --output Track01876_S00.mid
```

## `results/characterization_v3/per_sample.csv`

Written by `src/characterize.py`. Signal traits are measured on the audio as given to the model (mono, 22,050 Hz); content traits on the reference MIDI.

| Column | Meaning | Unit |
|---|---|---|
| `sample`, `split`, `inst_class` | As in `corpus.csv` | – |
| `program` | General MIDI program name of the stem, from Slakh metadata | – |
| `plugin` | Sampled instrument patch Slakh rendered the stem with | – |
| `loudness_lufs` | Integrated loudness, from Slakh metadata | LUFS |
| `format`, `subtype`, `source_rate`, `channels` | Container, PCM subtype, sample rate and channels of the file as stored | –, –, Hz, count |
| `duration_s` | Audio duration | s |
| `bitrate_kbps` | File size × 8 / duration | kbit/s |
| `peak_dbfs` | Highest absolute sample level | dBFS |
| `active_ratio` | Share of 2,048-sample frames whose RMS level is at least −60 dBFS | 0–1 |
| `active_rms_dbfs` | RMS level over those active frames | dBFS |
| `centroid_hz` | Median spectral centroid over the active frames | Hz |
| `notes` | Reference notes | count |
| `span_s` | First onset to last note-off | s |
| `density` | `notes` / `span_s` | notes/s |
| `attacks_per_s` | Attacks per second; an attack groups notes whose onsets fall within 30 ms of the group's first | attacks/s |
| `polyphony_mean` | Mean number of notes sounding, over the time any note sounds | notes |
| `polyphony_max` | Largest number of notes sounding at once | notes |
| `chord_share` | Share of notes that belong to an attack of two or more notes | 0–1 |
| `pitch_min`, `pitch_max`, `pitch_span` | Lowest and highest MIDI note number, and their difference | MIDI note, semitones |
| `duration_median_s`, `duration_median_beats` | Median note duration, in seconds and in beats at the sample's tempo | s, beats |
| `ioi_median_s` | Median time between consecutive attacks | s |
| `velocity_mean`, `velocity_sd` | Mean and population standard deviation of MIDI velocity | 0–127 |
| `tempo`, `grid` | As in `corpus.csv` | BPM, code |
| `grid_error_ratio` | RMS distance between the reference onsets and their grid lines, divided by the grid unit | grid units |
| `repetition_pct` | Share of the reference notation's tokens removed by repeat detection | % |

`summary_test.json` holds, for every instrument class and for all test samples (`All`): `n`, then under `numeric` the mean, standard deviation, median, minimum and maximum of each trait, and under `programs`, `plugins`, `format` and `grid` how often each value occurs. `level_boundaries` gives the two tercile boundaries of `span_s`, `density` and `repetition_pct` over all 1,575 samples: the values at zero-based positions ⌊n/3⌋ and ⌊2n/3⌋ of the sorted list, i.e. the lowest value of the medium and of the high level.

## `results/analysis_v3/`

Written by `src/analysis.py`. Samples of one song share tempo, arrangement and repeated sections, so every test treats the song as the unit: sample values are averaged per song (or per song and class) first. All 437 p-values, the 305 in `tests.csv` and the 132 in `instrument_pairs.csv`, were corrected together with the Benjamini–Hochberg procedure.

`tests.csv`:

| Column | Meaning |
|---|---|
| `question` | What the test asks |
| `test` | `Wilcoxon signed-rank`, `Kruskal-Wallis` or `Spearman` |
| `units` | Songs for Wilcoxon and Spearman; (song, class) units for Kruskal–Wallis |
| `before`, `after` | Wilcoxon only: mean over songs of the two conditions |
| `mean_difference`, `median_difference` | Wilcoxon only: mean and median of the per-song differences (after − before) |
| `up`, `down` | Wilcoxon only: songs whose value rose or fell |
| `groups`, `H`, `epsilon_squared` | Kruskal–Wallis only: number of classes, the H statistic, and ε² = (H − k + 1)/(n − k), the share of rank variance between classes |
| `scope`, `trait`, `outcome`, `rho` | Spearman only: `semua kelas` (all classes) or one class; a column of `characterization_v3/per_sample.csv`; `F-measure` or `S_token` (`saving_token_scheme_pct`); and Spearman's ρ |
| `p`, `q`, `significant` | Raw p-value, Benjamini–Hochberg q-value, and q ≤ 0.05 |

`instrument_pairs.csv`, one Mann–Whitney U test per pair of classes and outcome:

| Column | Meaning |
|---|---|
| `outcome` | `F-measure` (0–1) or `S_token` (%) |
| `first`, `second` | The two classes, in alphabetical order |
| `median_first`, `median_second` | Median over each class's (song, class) units |
| `superiority` | U / (n₁n₂): the chance that a unit of the first class exceeds one of the second |
| `p`, `q`, `significant` | As in `tests.csv` |

## `results/ablation_v3/`

Written by `src/ablation.py`. Each variant removes or replaces one component and is compared with the full system on the same samples.

| Variant | Layer | What changes |
|---|---|---|
| `full` | – | The full system; must reproduce `per_sample.csv` |
| `no_lz77` | Token | Repeat detection off |
| `absolute_positions` | Token | Absolute positions instead of delta encoding, field width fitted per sample |
| `byte_aligned` | Token | Header and tokens rounded up to whole bytes |
| `deflate`, `lzma` | Token | DEFLATE or LZMA over the token bytes instead of token-level LZ77 |
| `lz77_then_lzma` | Token | LZMA applied after token-level LZ77 |
| `midi_lzma` | Token | Tokens replaced by the transcribed MIDI file compressed with LZMA |
| `rs_shortened` | DNA | Zero padding in the last codeword not synthesised |
| `no_rs` | DNA | No Reed–Solomon outer code |
| `per_byte_trits` | DNA | 8 bits to 6 trits instead of 128-bit blocks |
| `naive_mapping` | DNA | 2 bits per base instead of the rotating code |
| `no_scrambling` | DNA | No scrambling before GC screening |
| `no_quantization` | Transcription | Raw transcription stored as MIDI |
| `fixed_grid` | Transcription | 1/16 grid for every sample |
| `notated_tempo` | Transcription | Written tempo and 1/16 grid; only for the 633 samples from constant-tempo stems |
| `default_onset` | Transcription | basic-pitch's default onset threshold, 0.5 |
| `default_min_note` | Transcription | basic-pitch's default minimum note length, 127.7 ms |

`ablation_per_sample.csv`:

| Column | Meaning | Unit |
|---|---|---|
| `sample`, `inst_class`, `pengulangan`, `variant` | Sample, class, repetition level, variant | – |
| `payload_bytes`, `bases` | Payload and bases under the variant | bytes, bases |
| `f_measure`, `f_measure_offset` | Accuracy under the variant | 0–1 |
| `delta_overflows` | Onset gaps above 255 grid units, which the 8-bit field cannot hold. They are counted and scored as if they could be stored | count |
| `oligos`, `gc_out_of_range`, `homopolymer_violations`, `max_homopolymer`, `rescrambles` | DNA-layer checks: oligos, oligos outside 40–60% GC, oligos with a homopolymer longer than 3, the longest homopolymer, and re-scrambles. Empty for variants that leave the DNA layer unchanged | count, bases |

`ablation_summary.csv`:

| Column | Meaning |
|---|---|
| `variant`, `label`, `family` | Variant code, Indonesian label, and `baseline`, `payload` (lossless layers) or `fidelity` (lossy layers) |
| `samples`, `tracks` | Samples and songs the variant ran on |
| `payload_total`, `bases_total` | Totals over those samples |
| `payload_change_pct`, `bases_change_pct` | (variant total / full-system total on the same samples − 1) × 100 |
| `payload_change_median_pct`, `bases_change_median_pct` | Median of the per-sample changes, in % |
| `more_bases`, `fewer_bases` | Samples that need more or fewer bases than the full system |
| `f_mean`, `f_change_mean` | Mean F-measure, and its mean difference from the full system |
| `p_payload`, `p_bases`, `p_f` | Wilcoxon signed-rank on per-song means against the full system; `p_f` for the `fidelity` family only |
| `q_payload`, `q_bases`, `q_f` | Benjamini–Hochberg q-values, corrected within each outcome |
| `delta_overflows`, `oligos`, `gc_out_of_range`, `homopolymer_violations`, `rescrambles`, `max_homopolymer` | Pooled over the samples, as in the per-sample file |

## Reproducing the files

The commands that produce every file, in order, are in the repository README under "Reproducing the experiments". The archived files were produced by the commit recorded in the archive's `README.md`.
