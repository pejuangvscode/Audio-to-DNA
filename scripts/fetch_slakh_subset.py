#!/usr/bin/env python3
"""
fetch_slakh_subset.py — ambil sebagian kecil Slakh2100 dari mirror Kaggle,
per file, tanpa mengunduh arsip 107 GB.

Arsip tar Slakh2100 diurut acak (entri train/test/validation/omitted saling
berselang-seling), jadi streaming partial extract tidak mungkin. Tapi Kaggle
melayani file individual, dan kagglehub.dataset_download() menerima argumen
`path` untuk satu file. Skrip ini memanfaatkan itu:

  1. tarik metadata.yaml tiap track (beberapa KB saja)
  2. baca, pilih stem bernada dari kelas instrumen yang diinginkan: satu per
     track, atau semuanya dengan --all-stems
  3. tarik HANYA stems/SXX.flac + MIDI/SXX.mid + all_src.mid untuk stem itu
  4. susun ulang jadi layout Slakh asli, supaya src.slakh jalan apa adanya

Satu stem rata-rata ~3,5 MB (mono, dan FLAC memampatkan bagian hening), jadi
seluruh 1.388 stem bernada di split test hanya ~4,6 GB, bukan 107,88 GB untuk
seluruh arsip. kagglehub menyimpan salinan cache sendiri, jadi pemakaian disk
sebenarnya kira-kira dua kali lipat.

Pakai, dari root repo:
  pip install kagglehub pyyaml

  # satu stem piano atau gitar per track (korpus v2)
  python scripts/fetch_slakh_subset.py --out new_dataset/slakh_subset --n 151

  # semua stem bernada dari seluruh split test (korpus v3)
  python scripts/fetch_slakh_subset.py --out new_dataset/slakh_subset --n 1000 --inst all --all-stems

Stem yang sudah ada di folder tujuan tidak disalin ulang.

Butuh kredensial Kaggle (kaggle.json di ~/.kaggle/, atau variabel lingkungan
KAGGLE_USERNAME dan KAGGLE_KEY). kagglehub menyimpan cache sendiri, jadi
menjalankan ulang tidak mengunduh dua kali.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

try:
    import kagglehub
    import yaml
except ImportError as e:
    sys.exit(f"paket kurang ({e.name}):  pip install kagglehub pyyaml")

HANDLE = "alonhaviv/slakh2100"
BASE = "slakh2100_flac_redux"


def grab(remote: str):
    """Tarik satu file dari dataset. Balikkan Path lokal, atau None kalau tidak ada."""
    try:
        return Path(kagglehub.dataset_download(HANDLE, path=remote))
    except Exception:
        return None


def pick_stems(meta: dict, want: set, verbose: bool = False) -> list[tuple[str, str]]:
    """Semua stem bernada yang benar-benar dirender dan MIDI-nya tersimpan,
    urut menurut id. `want` kosong berarti semua kelas diterima.
    """
    stems = meta.get("stems") or {}
    chosen = []
    for sid, s in sorted(stems.items()):
        if s.get("is_drum"):
            continue
        if s.get("audio_rendered") is False or s.get("midi_saved") is False:
            continue
        if want and s.get("inst_class") not in want:
            continue
        chosen.append((sid, s.get("inst_class", "?")))
    if not chosen and verbose and stems:
        seen = {s.get("inst_class") for s in stems.values()}
        print(f"       (kelas tersedia: {sorted(c for c in seen if c)})")
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="folder tujuan")
    ap.add_argument("--split", default="test",
                    choices=["test", "validation", "train", "omitted"])
    ap.add_argument("--n", type=int, default=30, help="berapa track yang diambil")
    ap.add_argument("--first", type=int, default=1876,
                    help="ID track awal yang dipindai (test biasanya mulai ~1876)")
    ap.add_argument("--last", type=int, default=2100, help="ID track akhir")
    ap.add_argument("--inst", default="Piano,Guitar",
                    help="kelas instrumen yang diterima, dipisah koma; 'all' untuk semua")
    ap.add_argument("--all-stems", action="store_true",
                    help="ambil semua stem yang cocok per track, bukan hanya yang pertama")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # 'all' rather than an empty string, which PowerShell 5.1 drops on the way
    # to a native program
    if args.inst.strip().lower() in ("", "all", "*"):
        want = set()
    else:
        want = {w.strip() for w in args.inst.split(",") if w.strip()}
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"memindai {args.split}/Track{args.first:05d}..Track{args.last:05d}")
    print(f"mencari {args.n} track, stem berkelas {sorted(want) or 'apa saja'}, "
          f"{'semua stem' if args.all_stems else 'satu stem'} per track\n")

    got, stems_got, skipped, scanned, t0 = 0, 0, 0, 0, time.time()
    total_bytes = 0

    for tid in range(args.first, args.last + 1):
        if got >= args.n:
            break
        track = f"Track{tid:05d}"
        scanned += 1

        meta_p = grab(f"{BASE}/{args.split}/{track}/metadata.yaml")
        if meta_p is None:
            continue  # track tidak ada di split/versi redux ini

        try:
            meta = yaml.safe_load(meta_p.read_text())
        except Exception:
            print(f"[!]  {track}: metadata.yaml tidak terbaca")
            continue

        chosen = pick_stems(meta, want, args.verbose)
        if not args.all_stems:
            chosen = chosen[:1]
        if not chosen:
            if args.verbose:
                print(f"[-]  {track}: tidak ada stem yang cocok")
            continue

        d = args.out / track
        (d / "stems").mkdir(parents=True, exist_ok=True)
        (d / "MIDI").mkdir(parents=True, exist_ok=True)
        shutil.copy(meta_p, d / "metadata.yaml")
        src = grab(f"{BASE}/{args.split}/{track}/all_src.mid")  # sumber tempo
        if src is not None:
            shutil.copy(src, d / "all_src.mid")

        taken = 0
        for sid, cls in chosen:
            flac_out = d / "stems" / f"{sid}.flac"
            midi_out = d / "MIDI" / f"{sid}.mid"
            if flac_out.is_file() and midi_out.is_file():
                skipped += 1
                taken += 1
                continue

            flac = grab(f"{BASE}/{args.split}/{track}/stems/{sid}.flac")
            midi = grab(f"{BASE}/{args.split}/{track}/MIDI/{sid}.mid")
            if flac is None or midi is None:
                print(f"[!]  {track}/{sid}: audio atau MIDI tidak ada, dilewati")
                continue
            shutil.copy(flac, flac_out)
            shutil.copy(midi, midi_out)

            sz = flac.stat().st_size
            total_bytes += sz
            stems_got += 1
            taken += 1
            warn = "" if src is not None else "   !! all_src.mid TIDAK ADA — tempo tak terverifikasi"
            print(f"[{got + 1:3d}/{args.n}] {track}/{sid:4s} {cls:22s} {sz/1e6:6.1f} MB{warn}")

        if taken:
            got += 1

    dt = time.time() - t0
    print(f"\n{'='*60}")
    print(f"{got} track dari {scanned} yang dipindai; {stems_got} stem baru diunduh, "
          f"{skipped} stem sudah ada")
    print(f"total {total_bytes/1e6:.0f} MB dalam {dt/60:.1f} menit  ->  {args.out}")

    if got == 0:
        print("\nTidak dapat apa-apa. Yang perlu dicek:")
        print("  - kredensial Kaggle (kaggle.json atau KAGGLE_USERNAME/KAGGLE_KEY)")
        print("  - rentang --first/--last: ID track di split ini mungkin berbeda")
        print("  - coba --inst all agar semua kelas instrumen diterima")
        return

    if got < args.n:
        print(f"\nSeluruh rentang Track{args.first:05d}..Track{args.last:05d} sudah dipindai; "
              f"hanya {got} track yang punya stem cocok.")
        print("Kalau kurang, lebarkan --first/--last atau longgarkan --inst.")

    print(f"\nLanjut:")
    print(f"  python -m src.slakh --source {args.out} --out "
          f"{'data/slakh_v3' if args.all_stems else 'data/slakh_v2'}")


if __name__ == "__main__":
    main()