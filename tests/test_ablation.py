"""Ablation variants (Subbab 3.3.5). Each variant must change only the part it
names, so the measured difference can be attributed to that part.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import config as cfg
from src import dna_codec as codec
from src.ablation import (
    NAIVE_PAYLOAD_NT,
    PER_BYTE_PAYLOAD_NT,
    VARIANTS,
    absolute_payload_bytes,
    absolute_position_tokens,
    base_count,
    benjamini_hochberg,
    build_oligos,
    byte_aligned_bytes,
    cached_transcription,
    compliance_counts,
    compressed_size,
    delta_overflows,
    naive_bases,
    paired_p,
    per_track,
    summarise,
)
from src.baselines import count_bases
from src.tokenizer import NoteToken, QuantizedNote, compress, encode_notes
from src.transcribe import TranscriptionParams


def repeated_phrase(times: int = 8) -> list[QuantizedNote]:
    """A one-bar figure played `times` times in a row at 1/16 resolution."""
    notes = []
    for bar in range(times):
        for step, pitch in enumerate((60, 64, 67, 64)):
            notes.append(QuantizedNote(pitch, bar * 16 + step * 4, 4))
    return notes


# --- payload sizes ---

def test_absolute_positions_defeat_repeat_detection():
    """Without deltas no two tokens repeat, so LZ77 finds nothing to refer to,
    even though the same figure is played eight times.
    """
    notes = repeated_phrase()
    tokens = compress(absolute_position_tokens(notes))

    assert all(isinstance(token, NoteToken) for token in tokens)
    assert absolute_payload_bytes(notes) > encode_notes(notes, 120).stats.payload_bytes


def test_byte_alignment_rounds_every_token_up_to_three_bytes():
    tokens = encode_notes(repeated_phrase(1), 120, use_repeat_detection=False).tokens
    assert byte_aligned_bytes(tokens) == 4 + 3 * len(tokens)


@pytest.mark.parametrize("method", ["deflate", "lzma"])
def test_general_compressors_round_trip_and_shrink_repetition(method):
    data = bytes(range(32)) * 40
    assert compressed_size(data, method) < len(data)


# --- base counts ---

def test_padded_count_matches_equation_3_2():
    for payload in (1, 142, 223, 224, 1000):
        assert base_count(payload) == count_bases(payload)[2]


def test_outer_code_variants_on_a_short_payload():
    """142 bytes is the case of Gambar 2.7: the padded codeword costs 255 bytes."""
    assert base_count(142) == 9 * cfg.OLIGO_TOTAL_NT  # 255 coded bytes
    assert base_count(142, outer="shortened") == 6 * cfg.OLIGO_TOTAL_NT  # 174
    assert base_count(142, outer="none") == 5 * cfg.OLIGO_TOTAL_NT  # 142


def test_shortening_changes_nothing_on_a_full_codeword():
    assert base_count(223, outer="shortened") == base_count(223)


def test_other_inner_codes_change_only_the_muatan_length():
    assert PER_BYTE_PAYLOAD_NT == 192
    assert NAIVE_PAYLOAD_NT == 128
    oligos = count_bases(500)[1]
    assert base_count(500, payload_nt=PER_BYTE_PAYLOAD_NT) == oligos * (192 + 40)


# --- inner code variants ---

def test_default_build_reproduces_the_codec():
    data = encode_notes(repeated_phrase(), 120).data
    oligos, rescrambles = build_oligos(data)
    reference = codec.encode(data)
    assert oligos == list(reference.oligos)
    assert rescrambles == reference.rescramble_count


def test_naive_mapping_uses_two_bits_per_base():
    assert naive_bases(bytes([0b00011011])) == "ACGT"
    oligos, _ = build_oligos(bytes(300), naive_bases)
    assert {len(oligo) for oligo in oligos} == {NAIVE_PAYLOAD_NT + 2 * cfg.PRIMER_NT}


def test_naive_mapping_breaks_the_homopolymer_limit():
    """Scrambled data mapped two bits per base still forms long runs somewhere."""
    oligos, rescrambles = build_oligos(bytes(2000), naive_bases)
    counts = compliance_counts(oligos, rescrambles)
    assert counts["max_homopolymer"] > cfg.HOMOPOLYMER_MAX
    assert counts["homopolymer_violations"] > 0


def test_unscrambled_rotating_code_still_has_no_long_runs():
    oligos, _ = build_oligos(bytes(2000), scramble=False)
    assert max(codec.max_homopolymer(oligo) for oligo in oligos) <= cfg.EXPECTED_MAX_HOMOPOLYMER


# --- fidelity helpers ---

def test_delta_overflow_is_counted():
    notes = [QuantizedNote(60, 0, 4), QuantizedNote(62, 300, 4), QuantizedNote(64, 310, 4)]
    assert delta_overflows(notes) == 1


def test_cached_transcription_is_read_without_running_the_model(tmp_path: Path):
    params = TranscriptionParams(onset_threshold=0.7)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / f"S1.prepared.abc.{params.fingerprint()}.json").write_text(
        json.dumps({"notes": [[60, 0.5, 1.0, 0.8]]})
    )

    notes = cached_transcription(tmp_path, "S1", params)
    assert notes is not None and notes[0].pitch == 60
    # another parameter set is not cached, and the model must not run
    assert cached_transcription(tmp_path, "S1", TranscriptionParams(onset_threshold=0.5)) is None


def test_cached_copies_are_read_when_they_agree_and_refused_when_not(tmp_path: Path):
    params = TranscriptionParams(onset_threshold=0.7)
    cache = tmp_path / "cache"
    cache.mkdir()
    for digest in ("old", "new"):
        (cache / f"S1.prepared.{digest}.{params.fingerprint()}.json").write_text(
            json.dumps({"notes": [[60, 0.5, 1.0, 0.8]]})
        )
    assert cached_transcription(tmp_path, "S1", params)[0].pitch == 60

    (cache / f"S1.prepared.other.{params.fingerprint()}.json").write_text(
        json.dumps({"notes": [[62, 0.5, 1.0, 0.8]]})
    )
    with pytest.raises(RuntimeError, match="disagree"):
        cached_transcription(tmp_path, "S1", params)


# --- statistics ---

def test_benjamini_hochberg_matches_a_hand_computed_example():
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    assert adjusted == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_identical_pairs_give_p_of_one():
    assert paired_p([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_summary_compares_each_variant_with_the_full_system():
    rows = []
    for index in range(6):
        sample = f"s{index}"
        rows.append({"sample": sample, "variant": "full", "payload_bytes": 100,
                     "bases": 1000, "f_measure": 0.8, "delta_overflows": 0})
        rows.append({"sample": sample, "variant": "no_lz77", "payload_bytes": 150 + index,
                     "bases": 1000, "f_measure": 0.8, "delta_overflows": 0})
    summary = {line["variant"]: line for line in summarise(rows)}

    # a payload change that stays inside one codeword leaves the bases alone
    assert summary["no_lz77"]["payload_change_pct"] > 50
    assert summary["no_lz77"]["bases_change_pct"] == 0
    assert summary["no_lz77"]["p_bases"] == 1.0
    assert summary["no_lz77"]["p_f"] == ""  # lossless variants do not test F
    assert summary["no_lz77"]["q_payload"] == summary["no_lz77"]["p_payload"]


def test_samples_of_one_track_count_as_one_unit():
    """Two stems of Track00001 and one of Track00002 give two track means."""
    samples = ["Track00001_S00", "Track00001_S01", "Track00002_S00"]
    assert per_track(samples, [10.0, 20.0, 7.0]) == [15.0, 7.0]

    rows = []
    for sample in samples:
        rows.append({"sample": sample, "variant": "full", "payload_bytes": 100,
                     "bases": 1000, "f_measure": 0.8, "delta_overflows": 0})
        rows.append({"sample": sample, "variant": "no_lz77", "payload_bytes": 150,
                     "bases": 1000, "f_measure": 0.8, "delta_overflows": 0})
    line = {l["variant"]: l for l in summarise(rows)}["no_lz77"]
    assert (line["samples"], line["tracks"]) == (3, 2)


def test_variant_keys_are_unique():
    keys = [variant.key for variant in VARIANTS]
    assert len(keys) == len(set(keys))
