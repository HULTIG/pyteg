"""Step 1. Download the raw data and check that the bytes are the expected ones.

Usage:  python pipeline/step1_download.py [--force]

The archives come from the nltk_data repository on GitHub, pinned to one
commit (see pipeline/sources.json). Each file is verified with SHA-256 and
then unpacked into data/raw/<name>/.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, ensure_dirs, load_sources  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(archive: Path, target: Path) -> None:
    """Unpack a zip file, refusing members that would land outside ``target``."""
    target = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            dest = (target / member.filename).resolve()
            if target not in dest.parents and dest != target:
                raise RuntimeError(f"unsafe path in {archive.name}: {member.filename}")
        zf.extractall(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="download again even if the file exists")
    args = parser.parse_args()

    ensure_dirs()
    sources = load_sources()
    archives_dir = RAW / "archives"
    archives_dir.mkdir(parents=True, exist_ok=True)

    for name, info in sources["archives"].items():
        archive = archives_dir / f"{name}.zip"
        if args.force or not archive.exists() or sha256(archive) != info["sha256"]:
            url = sources["base_url"] + f"{name}.zip"
            print(f"downloading {name} ({info['use']})")
            with urllib.request.urlopen(url, timeout=300) as response, open(archive, "wb") as out:
                shutil.copyfileobj(response, out)
        digest = sha256(archive)
        if digest != info["sha256"]:
            print(f"checksum mismatch for {name}: expected {info['sha256']}, got {digest}", file=sys.stderr)
            return 1
        print(f"ok  {name:<11} {digest[:16]}...")
        if not (RAW / name).exists() or args.force:
            safe_extract(archive, RAW)
    print(f"raw data is in {RAW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
