"""Ablation study: take one design decision out at a time and measure what it
was worth (Subbab 3.3.5).

Every variant starts from the transcription the main experiment already cached,
so basic-pitch only runs again for the two variants that change its parameters,
and only when --transcribe is given.

Variants fall into two families:

    payload    the token scheme and the DNA layer. All lossless, so they change
               the base count, and for the inner code also compliance, but never
               the reconstructed notes.
    fidelity   transcription and quantisation. Lossy, so they change the
               F-measure as well as the base count.

Each variant is compared with the full system on the same samples by a Wilcoxon
signed-rank test, with Benjamini-Hochberg applied within each outcome (Subbab
2.9.5).

    python -m src.ablation --data-dir data/slakh_v2 --split test \
        --work-dir results/slakh_test_v2/work --onset-threshold 0.7 \
        --output results/ablation_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import lzma
import math
import statistics
import sys
import tempfile
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import pretty_midi

from src import config as cfg
from src import dna_codec as codec
from src import evaluate
from src.corpus import CorpusSample, load_manifest
from src.quantize import QuantizationReport, quantize
from src.reconstruct import read_midi, to_midi
from src.tokenizer import NoteToken, QuantizedNote, Token, compress, encode_notes, notes_to_tokens
from src.transcribe import TranscribedNote, TranscriptionParams, transcribe_cached

# basic-pitch's own defaults (Tabel 2.5), which Subbab 3.3.2 departs from
LIBRARY_ONSET_THRESHOLD = 0.5
LIBRARY_MINIMUM_NOTE_LENGTH_MS = 127.70

# muatan length when each byte is converted alone: 8 bits need 6 trits (Tabel 2.2)
PER_BYTE_PAYLOAD_NT = cfg.OLIGO_PAYLOAD_BYTES * math.ceil(8 / math.log2(3))

# muatan length under the naive mapping, two bits per base (Subbab 2.3.1)
NAIVE_PAYLOAD_NT = cfg.OLIGO_PAYLOAD_BYTES * 4


@dataclass(frozen=True)
class Variant:
    key: str
    label: str  # as the report names it
    family: str  # "baseline", "payload" or "fidelity"


VARIANTS = (
    Variant("full", "Sistem lengkap", "baseline"),
    Variant("no_lz77", "Tanpa deteksi pengulangan", "payload"),
    Variant("absolute_positions", "Tanpa pengodean selisih (posisi mutlak)", "payload"),
    Variant("byte_aligned", "Tanpa pemadatan bit (token selaras byte)", "payload"),
    Variant("deflate", "LZ77 token diganti DEFLATE atas byte token", "payload"),
    Variant("lzma", "LZ77 token diganti LZMA atas byte token", "payload"),
    Variant("lz77_then_lzma", "LZ77 token ditambah LZMA", "payload"),
    Variant("midi_lzma", "Skema token diganti MIDI transkripsi + LZMA", "payload"),
    Variant("rs_shortened", "Reed–Solomon diperpendek (pengisi tidak disintesis)", "payload"),
    Variant("no_rs", "Tanpa Reed–Solomon", "payload"),
    Variant("per_byte_trits", "Konversi trit per byte (8 bit menjadi 6 trit)", "payload"),
    Variant("naive_mapping", "Rotating code diganti pemetaan naif 2 bit/basa", "payload"),
    Variant("no_scrambling", "Tanpa pengacakan", "payload"),
    Variant("no_quantization", "Tanpa kuantisasi (keluaran transkripsi)", "fidelity"),
    Variant("fixed_grid", "Kisi tetap seperenam belas", "fidelity"),
    Variant("notated_tempo", "Tempo tertulis dan kisi seperenam belas", "fidelity"),
    Variant("default_onset", "Threshold onset bawaan (0,5)", "fidelity"),
    Variant("default_min_note", "Durasi not minimum bawaan (127,7 ms)", "fidelity"),
)

VARIANT_BY_KEY = {variant.key: variant for variant in VARIANTS}


# --- payload sizes ---

def absolute_position_tokens(notes: Sequence[QuantizedNote]) -> list[NoteToken]:
    """Note tokens whose position field holds the absolute onset, not the delta."""
    return [NoteToken(note.pitch, note.position, note.duration) for note in notes]


def absolute_payload_bytes(notes: Sequence[QuantizedNote]) -> int:
    """Packed size without delta encoding, repeat detection still on.

    The position field gets the fewest bits that hold this sample's last onset.
    A real format would have to fix that width in advance, so this flatters the
    variant, which makes the measured cost a lower bound.
    """
    tokens = compress(absolute_position_tokens(notes))
    width = max((note.position for note in notes), default=0).bit_length() or 1
    note_bits = cfg.TOKEN_TYPE_BITS + cfg.NOTE_PITCH_BITS + width + cfg.NOTE_DURATION_BITS
    bits = cfg.HEADER_BITS + sum(
        note_bits if isinstance(token, NoteToken) else cfg.REFERENCE_TOKEN_BITS
        for token in tokens
    )
    return -(-bits // 8)


def byte_aligned_bytes(tokens: Sequence[Token]) -> int:
    """Packed size when the header and every token start on a byte boundary."""
    note_bytes = -(-cfg.NOTE_TOKEN_BITS // 8)
    reference_bytes = -(-cfg.REFERENCE_TOKEN_BITS // 8)
    return -(-cfg.HEADER_BITS // 8) + sum(
        note_bytes if isinstance(token, NoteToken) else reference_bytes
        for token in tokens
    )


# Raw streams, without the container headers a small payload would pay for.
_LZMA_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]


def _deflate(data: bytes) -> bytes:
    packer = zlib.compressobj(9, zlib.DEFLATED, -15, 9)
    return packer.compress(data) + packer.flush()


COMPRESSORS: dict[str, tuple[Callable[[bytes], bytes], Callable[[bytes], bytes]]] = {
    # byte-level LZ77 plus Huffman coding
    "deflate": (_deflate, lambda data: zlib.decompress(data, -15)),
    # byte-level LZ77 plus range coding with context modelling
    "lzma": (
        lambda data: lzma.compress(data, format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS),
        lambda data: lzma.decompress(data, format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS),
    ),
}


def compressed_size(data: bytes, method: str) -> int:
    """Size after a general-purpose compressor, checked to decompress exactly."""
    pack, unpack = COMPRESSORS[method]
    packed = pack(data)
    if unpack(packed) != data:
        raise RuntimeError(f"{method} did not round-trip")
    return len(packed)


# --- base counts ---

def coded_length(payload_bytes: int, outer: str = "padded") -> int:
    """Bytes after the outer code.

    padded     the system as built: the last codeword is zero-filled and the
               fill is synthesised (Gambar 2.7)
    shortened  parity is kept but the fill is not synthesised, since the decoder
               knows it is zero
    none       no outer code at all
    """
    codewords = -(-payload_bytes // cfg.RS_K)
    if outer == "padded":
        return codewords * cfg.RS_N
    if outer == "shortened":
        return payload_bytes + codewords * cfg.RS_NSYM
    if outer == "none":
        return payload_bytes
    raise ValueError(f"unknown outer code {outer!r}")


def base_count(
    payload_bytes: int, outer: str = "padded", payload_nt: int = cfg.OLIGO_PAYLOAD_NT
) -> int:
    """Persamaan 3.2 with the outer code and the muatan length as parameters."""
    oligos = -(-coded_length(payload_bytes, outer) // cfg.OLIGO_DATA_BYTES)
    return oligos * (payload_nt + 2 * cfg.PRIMER_NT)


# --- inner code variants, built for real so compliance can be measured ---

def naive_bases(payload: bytes) -> str:
    """Two bits per base: 00 A, 01 C, 10 G, 11 T (Subbab 2.3.1)."""
    return "".join(
        cfg.BASES[(byte >> shift) & 3] for byte in payload for shift in (6, 4, 2, 0)
    )


def rotating_bases(payload: bytes) -> str:
    """The inner code as built (Subbab 3.3.4, tahap keempat dan kelima)."""
    return codec._payload_to_bases(payload)


def build_oligos(
    data: bytes,
    payload_to_bases: Callable[[bytes], str] = rotating_bases,
    scramble: bool = True,
) -> tuple[list[str], int]:
    """codec.encode() with the inner code and the scrambling swappable.

    GC screening stays in every variant, so only the ablated part changes. With
    the defaults this reproduces codec.encode() exactly. Returns the oligos and
    the number of rescrambles.
    """
    coded = codec.rs_encode(data)
    oligos: list[str] = []
    rescrambles = 0

    for index in range(-(-len(coded) // cfg.OLIGO_DATA_BYTES)):
        start = index * cfg.OLIGO_DATA_BYTES
        chunk = coded[start : start + cfg.OLIGO_DATA_BYTES].ljust(cfg.OLIGO_DATA_BYTES, b"\x00")
        for seed_counter in range(cfg.MAX_SCRAMBLE_ATTEMPTS):
            body = codec.scramble(chunk, index, seed_counter) if scramble else chunk
            payload = index.to_bytes(cfg.OLIGO_INDEX_BYTES, "big") + bytes([seed_counter]) + body
            oligo = cfg.PRIMER_FORWARD + payload_to_bases(payload) + cfg.PRIMER_REVERSE
            if cfg.GC_MIN <= codec.gc_fraction(oligo) <= cfg.GC_MAX:
                break
            rescrambles += 1
        oligos.append(oligo)

    return oligos, rescrambles


def compliance_counts(oligos: Sequence[str], rescrambles: int) -> dict:
    """Counts rather than fractions, so the summary can pool them over samples."""
    return {
        "oligos": len(oligos),
        "gc_out_of_range": sum(
            not cfg.GC_MIN <= codec.gc_fraction(oligo) <= cfg.GC_MAX for oligo in oligos
        ),
        "homopolymer_violations": sum(
            codec.max_homopolymer(oligo) > cfg.HOMOPOLYMER_MAX for oligo in oligos
        ),
        "max_homopolymer": max(codec.max_homopolymer(oligo) for oligo in oligos),
        "rescrambles": rescrambles,
    }


# --- fidelity helpers ---

def delta_overflows(notes: Sequence[QuantizedNote]) -> int:
    """Onset deltas too wide for the 8-bit field. The packer would wrap them
    silently, so the variant is scored on its notes and the overflow reported.
    """
    return sum(token.delta > cfg.MAX_DELTA_UNITS for token in notes_to_tokens(notes))


def cached_transcription(
    work_dir: Path, name: str, params: TranscriptionParams, transcribe_missing: bool = False
) -> tuple[TranscribedNote, ...] | None:
    """The transcription the experiment cached for these parameters, or None.

    Runs basic-pitch only when `transcribe_missing` is set, on the prepared WAV
    the experiment left behind, and never overwrites its transcribed MIDI.
    """
    cache_dir = work_dir / "cache"
    matches = sorted(cache_dir.glob(f"{name}.prepared.*.{params.fingerprint()}.json"))
    if matches:
        # one sample can hold several entries: before the cache keyed on the
        # decoded samples, every rerun added one. Copies that agree are harmless.
        found = {tuple(map(tuple, json.loads(path.read_text())["notes"])) for path in matches}
        if len(found) > 1:
            raise RuntimeError(f"{len(matches)} cached transcriptions of {name} disagree; clear the stale ones")
        return tuple(TranscribedNote(*note) for note in found.pop())

    prepared = work_dir / f"{name}.prepared.wav"
    if not transcribe_missing or not prepared.is_file():
        return None
    return transcribe_cached(prepared, cache_dir, params).notes


def notated_tempo(sample: CorpusSample, slakh_root: Path) -> int | None:
    """The tempo written in the source track's MIDI, before Subbab 3.2.5 fitted it.

    None for a stem whose tempo changes: its segments come from one tempo region
    each, and the manifest does not record which, so the variant skips them.
    """
    from src.slakh import _constant_tempo

    track, stem = sample.name.split("_")[:2]
    midi_path = slakh_root / track / "MIDI" / f"{stem}.mid"
    if not midi_path.is_file():
        return None
    return _constant_tempo(pretty_midi.PrettyMIDI(str(midi_path)))


def score(
    reference: Sequence, notes: Sequence[QuantizedNote], tempo: int, grid_code: int, scratch: Path
) -> evaluate.AccuracyScores:
    """Through a MIDI file, as the decoder writes one, so tick rounding matches
    the main experiment.
    """
    path = to_midi(notes, scratch / "ablation.mid", tempo, grid_code)
    return evaluate.accuracy(reference, read_midi(path))


# --- one sample ---

def run_sample(
    sample: CorpusSample,
    work_dir: Path,
    params: TranscriptionParams,
    slakh_root: Path,
    scratch: Path,
    transcribe_missing: bool = False,
) -> tuple[list[dict], list[str]]:
    """Every variant for one sample, as one row each."""
    rows: list[dict] = []
    problems: list[str] = []

    notes = cached_transcription(work_dir, sample.name, params, transcribe_missing)
    if notes is None:
        return rows, [
            f"no cached transcription at onset {params.onset_threshold}, minimum note "
            f"{params.minimum_note_length_ms} ms; run src.experiment with these "
            "parameters first"
        ]

    reference = read_midi(sample.midi_path)
    tempo, grid = sample.tempo, sample.grid_code

    full_q = quantize(notes, tempo, grid)
    full_tokens = encode_notes(full_q.notes, tempo, grid)
    full_dna = codec.encode(full_tokens.data)
    full_scores = score(reference, full_q.notes, tempo, grid, scratch)
    payload = full_tokens.stats.payload_bytes

    def add(
        key: str,
        payload_bytes: int,
        bases: int,
        scores: evaluate.AccuracyScores = full_scores,
        compliance: dict | None = None,
        overflows: int = 0,
    ) -> None:
        row = {
            "sample": sample.name,
            "inst_class": sample.inst_class,
            "pengulangan": sample.repetition_level,
            "variant": key,
            "payload_bytes": payload_bytes,
            "bases": bases,
            "f_measure": round(scores.f_measure, 4),
            "f_measure_offset": round(scores.f_measure_with_offset, 4),
            "delta_overflows": overflows,
        }
        row.update(compliance or {})
        rows.append(row)

    def add_quantised(key: str, quantised: QuantizationReport, q_tempo: int, q_grid: int) -> None:
        tokens = encode_notes(quantised.notes, q_tempo, q_grid)
        add(
            key,
            tokens.stats.payload_bytes,
            base_count(tokens.stats.payload_bytes),
            scores=score(reference, quantised.notes, q_tempo, q_grid, scratch),
            overflows=delta_overflows(quantised.notes),
        )

    # --- payload family ---
    add("full", payload, full_dna.total_bases,
        compliance=compliance_counts(full_dna.oligos, full_dna.rescramble_count))

    plain = encode_notes(full_q.notes, tempo, grid, use_repeat_detection=False)
    add("no_lz77", plain.stats.payload_bytes, base_count(plain.stats.payload_bytes))

    absolute = absolute_payload_bytes(full_q.notes)
    add("absolute_positions", absolute, base_count(absolute))

    aligned = byte_aligned_bytes(full_tokens.tokens)
    add("byte_aligned", aligned, base_count(aligned))

    for key, method, data in (
        ("deflate", "deflate", plain.data),
        ("lzma", "lzma", plain.data),
        ("lz77_then_lzma", "lzma", full_tokens.data),
    ):
        size = compressed_size(data, method)
        add(key, size, base_count(size))

    add("rs_shortened", payload, base_count(payload, outer="shortened"))
    add("no_rs", payload, base_count(payload, outer="none"))
    add("per_byte_trits", payload, base_count(payload, payload_nt=PER_BYTE_PAYLOAD_NT))

    for key, to_bases, scramble in (
        ("naive_mapping", naive_bases, True),
        ("no_scrambling", rotating_bases, False),
    ):
        oligos, rescrambles = build_oligos(full_tokens.data, to_bases, scramble)
        add(key, payload, sum(map(len, oligos)), compliance=compliance_counts(oligos, rescrambles))

    # --- fidelity family ---
    transcribed_midi = work_dir / f"{sample.name}.transcribed.mid"
    if transcribed_midi.is_file():
        # what B2 encodes: the model's own MIDI, unquantised and untokenised
        midi_bytes = transcribed_midi.read_bytes()
        raw_scores = evaluate.accuracy(reference, read_midi(transcribed_midi))
        add("no_quantization", len(midi_bytes), base_count(len(midi_bytes)), scores=raw_scores)
        size = compressed_size(midi_bytes, "lzma")
        add("midi_lzma", size, base_count(size), scores=raw_scores)
    else:
        problems.append(f"{transcribed_midi} is missing; skipped the unquantised variants")

    add_quantised("fixed_grid", quantize(notes, tempo, cfg.DEFAULT_GRID_CODE), tempo, cfg.DEFAULT_GRID_CODE)

    notated = notated_tempo(sample, slakh_root)
    if notated is not None and cfg.MIN_TEMPO <= notated <= cfg.MAX_TEMPO:
        add_quantised(
            "notated_tempo",
            quantize(notes, notated, cfg.DEFAULT_GRID_CODE),
            notated,
            cfg.DEFAULT_GRID_CODE,
        )
    else:
        problems.append("notated tempo unavailable; skipped notated_tempo")

    for key, alternative in (
        ("default_onset", replace(params, onset_threshold=LIBRARY_ONSET_THRESHOLD)),
        ("default_min_note", replace(params, minimum_note_length_ms=LIBRARY_MINIMUM_NOTE_LENGTH_MS)),
    ):
        alt_notes = cached_transcription(work_dir, sample.name, alternative, transcribe_missing)
        if alt_notes is None:
            problems.append(f"no transcription cached for {key}; rerun with --transcribe")
            continue
        add_quantised(key, quantize(alt_notes, tempo, grid), tempo, grid)

    return rows, problems


# --- statistics ---

def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Adjusted p-values (Persamaan 2.24); a test is significant at FDR q when
    its adjusted value is at most q.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda index: p_values[index])
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        index = order[rank - 1]
        running = min(running, p_values[index] * m / rank)
        adjusted[index] = running
    return adjusted


def paired_p(variant: Sequence[float], full: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value; 1 when no pair differs."""
    if all(v == f for v, f in zip(variant, full)):
        return 1.0
    from scipy.stats import wilcoxon

    return float(wilcoxon(variant, full).pvalue)


def track_of(sample: str) -> str:
    """The Slakh track a sample comes from: Track01884_S02_s01 -> Track01884."""
    return sample.split("_")[0]


def per_track(samples: Sequence[str], values: Sequence[float]) -> list[float]:
    """Mean value per track, in track order.

    Stems and segments of one track share its song, tempo and repeated
    sections, so they are not independent observations; the paired tests run
    on these means, one per track.
    """
    groups: dict[str, list[float]] = {}
    for sample, value in zip(samples, values):
        groups.setdefault(track_of(sample), []).append(value)
    return [statistics.fmean(groups[track]) for track in sorted(groups)]


def paired_tracks(samples, variant, full) -> float:
    return paired_p(per_track(samples, variant), per_track(samples, full))


def summarise(rows: Sequence[dict]) -> list[dict]:
    """One line per variant against the full system on the samples both cover.
    Totals and medians are over samples; p-values are over tracks.
    """
    by_variant: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], {})[row["sample"]] = row
    full = by_variant.get("full", {})

    summary: list[dict] = []
    for variant in VARIANTS:
        got = by_variant.get(variant.key)
        if not got:
            continue
        samples = sorted(name for name in got if name in full)
        if not samples:
            continue
        payload = [got[name]["payload_bytes"] for name in samples]
        payload_full = [full[name]["payload_bytes"] for name in samples]
        bases = [got[name]["bases"] for name in samples]
        bases_full = [full[name]["bases"] for name in samples]
        f_values = [got[name]["f_measure"] for name in samples]
        f_full = [full[name]["f_measure"] for name in samples]

        # Reported at both levels because they can disagree: Reed-Solomon
        # codeword granularity hides payload changes that stay inside one
        # codeword (Gambar 2.7).
        line = {
            "variant": variant.key,
            "label": variant.label,
            "family": variant.family,
            "samples": len(samples),
            "tracks": len({track_of(name) for name in samples}),
            "payload_total": sum(payload),
            "payload_change_pct": round((sum(payload) / sum(payload_full) - 1) * 100.0, 2),
            "payload_change_median_pct": round(
                statistics.median((p / f - 1) * 100.0 for p, f in zip(payload, payload_full)), 2
            ),
            "p_payload": ""
            if variant.family == "baseline"
            else paired_tracks(samples, payload, payload_full),
            "bases_total": sum(bases),
            "bases_change_pct": round((sum(bases) / sum(bases_full) - 1) * 100.0, 2),
            "bases_change_median_pct": round(
                statistics.median((b / f - 1) * 100.0 for b, f in zip(bases, bases_full)), 2
            ),
            "more_bases": sum(b > f for b, f in zip(bases, bases_full)),
            "fewer_bases": sum(b < f for b, f in zip(bases, bases_full)),
            "p_bases": ""
            if variant.family == "baseline"
            else paired_tracks(samples, bases, bases_full),
            "f_mean": round(statistics.fmean(f_values), 4),
            "f_change_mean": round(statistics.fmean(f - g for f, g in zip(f_values, f_full)), 4),
            "p_f": paired_tracks(samples, f_values, f_full) if variant.family == "fidelity" else "",
            "delta_overflows": sum(got[name]["delta_overflows"] for name in samples),
        }
        if "oligos" in got[samples[0]]:
            pooled = {
                key: sum(got[name][key] for name in samples)
                for key in ("oligos", "gc_out_of_range", "homopolymer_violations", "rescrambles")
            }
            pooled["max_homopolymer"] = max(got[name]["max_homopolymer"] for name in samples)
            line.update(pooled)
        summary.append(line)

    # FDR is controlled within each outcome, as in the main analysis
    for outcome in ("payload", "bases", "f"):
        tested = [line for line in summary if line[f"p_{outcome}"] != ""]
        for line, q in zip(tested, benjamini_hochberg([line[f"p_{outcome}"] for line in tested])):
            line[f"q_{outcome}"] = q
    return summary


def cross_check(rows: Sequence[dict], per_sample_csv: Path) -> list[str]:
    """The full variant must reproduce the main experiment, or nothing measured
    against it means anything.
    """
    with open(per_sample_csv, newline="", encoding="utf-8") as handle:
        main_rows = {row["sample"]: row for row in csv.DictReader(handle)}

    checks = (
        ("full", "bases", "bases_pipeline", 0.0),
        ("full", "f_measure", "f_measure", 1e-4),
        ("no_lz77", "bases", "bases_no_lz77", 0.0),
        ("no_quantization", "f_measure", "f_transcription", 1e-4),
    )
    notes: list[str] = []
    for variant, ours, theirs, tolerance in checks:
        compared = mismatched = 0
        for row in rows:
            other = main_rows.get(row["sample"])
            if row["variant"] != variant or other is None or other.get(theirs, "") == "":
                continue
            compared += 1
            if abs(float(row[ours]) - float(other[theirs])) > tolerance:
                mismatched += 1
        notes.append(
            f"{variant}.{ours} vs experiment {theirs}: "
            f"{compared - mismatched} of {compared} samples agree"
        )
    return notes


# --- output ---

def _id(value: float, spec: str) -> str:
    """A number in Indonesian notation: decimal comma, thousands point."""
    return format(value, spec).replace(",", "_").replace(".", ",").replace("_", ".")


def _format_p(value) -> str:
    if value == "":
        return "–"
    return "< 0,001" if value < 0.001 else _id(value, ".3f")


def write_outputs(rows: Sequence[dict], summary: Sequence[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, table in (("ablation_per_sample.csv", rows), ("ablation_summary.csv", summary)):
        columns: list[str] = []
        for row in table:
            columns += [key for key in row if key not in columns]
        with open(out_dir / name, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(table)

    family_titles = {
        "payload": "Lapisan token dan DNA (lossless)",
        "fidelity": "Lapisan transkripsi dan kuantisasi (lossy)",
    }
    full = next((line for line in summary if line["variant"] == "full"), None)
    lines = ["# Hasil ablation study", ""]
    if full:
        lines += [
            f"Sistem lengkap: {_id(full['bases_total'], ',')} basa pada {full['samples']} "
            f"sampel, F-measure rerata {_id(full['f_mean'], '.3f')}.",
            "",
            "Δ payload (byte token) dan Δ basa dihitung atas total korpus terhadap sistem "
            "lengkap; median per sampel ada di ablation_summary.csv. q adalah nilai p "
            "Wilcoxon signed-rank atas rata-rata per track (stem dan potongan dari satu "
            "lagu tidak independen), setelah koreksi Benjamini–Hochberg dalam setiap luaran.",
        ]

    for family, title in family_titles.items():
        members = [line for line in summary if line["family"] == family]
        if not members:
            continue
        lines += ["", f"## {title}", ""]
        lines += [
            "| Varian | Sampel (track) | Δ payload % | q (payload) | Δ basa % | q (basa) "
            "| F rerata | ΔF | q (F) | Kepatuhan |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for line in members:
            compliance = "–"
            if "oligos" in line:
                compliance = (
                    f"GC di luar rentang {line['gc_out_of_range']}/{line['oligos']}; "
                    f"homopolymer > {cfg.HOMOPOLYMER_MAX}: "
                    f"{line['homopolymer_violations']}/{line['oligos']} "
                    f"(maks {line['max_homopolymer']})"
                )
            if line["delta_overflows"]:
                compliance = f"selisih posisi > 255: {line['delta_overflows']} not"
            lines.append(
                f"| {line['label']} | {line['samples']} ({line['tracks']}) "
                f"| {_id(line['payload_change_pct'], '+.1f')} "
                f"| {_format_p(line.get('q_payload', ''))} "
                f"| {_id(line['bases_change_pct'], '+.1f')} "
                f"| {_format_p(line.get('q_bases', ''))} "
                f"| {_id(line['f_mean'], '.3f')} | {_id(line['f_change_mean'], '+.3f')} "
                f"| {_format_p(line.get('q_f', ''))} | {compliance} |"
            )

    path = out_dir / "ablation_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.ablation",
        description="Remove one design decision at a time and measure its effect.",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="the work directory of the src.experiment run, with its transcription cache",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=cfg.ONSET_THRESHOLD,
        help="the calibrated value the experiment ran with (config: "
        f"{cfg.ONSET_THRESHOLD})",
    )
    parser.add_argument(
        "--slakh-root",
        type=Path,
        default=Path("new_dataset/slakh_subset"),
        help="source Slakh tracks, for the notated tempo",
    )
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help="run basic-pitch for the two variants that change its parameters",
    )
    parser.add_argument("--only", nargs="*", help="run only these samples, by name")
    parser.add_argument(
        "--check",
        type=Path,
        default=None,
        help="per_sample.csv of the main experiment (default: next to --work-dir)",
    )
    args = parser.parse_args(argv)

    params = TranscriptionParams(onset_threshold=args.onset_threshold)
    samples = [sample for sample in load_manifest(args.data_dir) if sample.split == args.split]
    if args.only:
        samples = [sample for sample in samples if sample.name in set(args.only)]

    rows: list[dict] = []
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as scratch:
        for index, sample in enumerate(samples, start=1):
            print(f"[{index}/{len(samples)}] {sample.name} ...", flush=True)
            sample_rows, sample_problems = run_sample(
                sample, args.work_dir, params, args.slakh_root, Path(scratch), args.transcribe
            )
            rows += sample_rows
            problems += [f"{sample.name}: {problem}" for problem in sample_problems]

    if not rows:
        print("nothing was measured", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    summary = summarise(rows)
    path = write_outputs(rows, summary, args.output)

    check = args.check or args.work_dir.parent / "per_sample.csv"
    print()
    if check.is_file():
        for note in cross_check(rows, check):
            print(note)
    print(f"tables written to {path}")
    if problems:
        print(f"{len(problems)} problems:")
        for problem in problems:
            print(f"  {problem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
