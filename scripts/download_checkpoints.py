#!/usr/bin/env python3
"""Downloads the trained checkpoints released alongside this repository from
Zenodo and extracts them into runs/<arch>_<dataset>/final/ -- the same layout
`cabrnet train` produces, so run_formal_exp.py works identically whether you
trained a checkpoint yourself or downloaded it.
"""
import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

# TODO: replace with the real Zenodo record ID once the checkpoints are published.
ZENODO_RECORD_ID = "XXXXXXX"
ARCHIVE_NAME = "beyond_l2_checkpoints.zip"

COMBOS = [
    ("protopnet", "cub200"), ("protopnet", "flowers102"), ("protopnet", "oxford_iiit_pet"),
    ("tesnet", "cub200"), ("tesnet", "flowers102"), ("tesnet", "oxford_iiit_pet"),
    ("pipnet", "cub200"), ("pipnet", "flowers102"), ("pipnet", "oxford_iiit_pet"),
    ("protopool", "cub200"),
    ("protopnet_gaussian_iso", "cub200"), ("protopnet_gaussian_iso", "flowers102"),
    ("protopnet_gaussian_iso", "oxford_iiit_pet"),
]


def fetch_record_metadata(record_id):
    url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)


def download_with_progress(url, dest):
    def report(block_num, block_size, total_size):
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        pct = done * 100 // total_size
        sys.stdout.write(f"\r  {pct:3d}% ({done // (1024*1024)} / {total_size // (1024*1024)} MB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=report)
    print()


def hash_file(path, algo="sha256", chunk_size=1024 * 1024):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-id", default=ZENODO_RECORD_ID,
                         help="Zenodo record ID to download from (default: the one released with this repo)")
    parser.add_argument("--dest-root", type=Path, default=Path("runs"),
                         help="Where to extract <arch>_<dataset>/final/ into (default: runs/)")
    parser.add_argument("--only", nargs="*", metavar="ARCH_DATASET",
                         help="Only extract these combos, e.g. --only tesnet_flowers102 protopnet_cub200")
    parser.add_argument("--keep-zip", action="store_true", help="Don't delete the downloaded archive afterwards")
    args = parser.parse_args()

    if args.record_id == "XXXXXXX":
        sys.exit("error: no Zenodo record ID configured yet -- checkpoints haven't been published.")

    print(f"Fetching record metadata for Zenodo record {args.record_id}...")
    record = fetch_record_metadata(args.record_id)
    files = {f["key"]: f for f in record["files"]}
    if ARCHIVE_NAME not in files:
        sys.exit(f"error: {ARCHIVE_NAME} not found in Zenodo record {args.record_id}")
    file_info = files[ARCHIVE_NAME]

    archive_path = Path(ARCHIVE_NAME)
    print(f"Downloading {ARCHIVE_NAME} ({file_info['size'] / (1024*1024):.0f} MB)...")
    download_with_progress(file_info["links"]["self"], archive_path)

    algo, expected = file_info["checksum"].split(":", 1)
    print("Verifying checksum...")
    actual = hash_file(archive_path, algo)
    if actual != expected:
        archive_path.unlink()
        sys.exit(f"error: checksum mismatch (expected {expected}, got {actual}) -- deleted the download")
    print("Checksum OK.")

    wanted = set(args.only) if args.only else {f"{a}_{d}" for a, d in COMBOS}

    print(f"Extracting to {args.dest_root}/...")
    with zipfile.ZipFile(archive_path) as zf:
        for arch, dataset in COMBOS:
            key = f"{arch}_{dataset}"
            if key not in wanted:
                continue
            prefix = f"{arch}/{dataset}/final/"
            members = [m for m in zf.namelist() if m.startswith(prefix)]
            if not members:
                print(f"warning: no files found for {key} in the archive")
                continue
            out_dir = args.dest_root / key / "final"
            out_dir.mkdir(parents=True, exist_ok=True)
            for member in members:
                (out_dir / Path(member).name).write_bytes(zf.read(member))
            print(f"  {key}")

    if not args.keep_zip:
        archive_path.unlink()

    print("Done.")


if __name__ == "__main__":
    main()
