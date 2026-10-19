"""Download the pinned official arXiv source and extract regular files safely."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
URL = "https://arxiv.org/src/2510.21890v3"
DEST = ROOT / "source/arxiv-2510.21890v3"


def unpack(archive: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        if sum(m.size for m in members) > 250_000_000:
            raise ValueError("Unexpectedly large source archive")
        for member in members:
            name = PurePosixPath(member.name)
            target = (root / member.name).resolve()
            if (name.is_absolute() or ".." in name.parts or "\\" in member.name
                    or ":" in member.name or not target.is_relative_to(root)
                    or not (member.isfile() or member.isdir())):
                raise ValueError(f"Unsafe archive member: {member.name}")
        # data filter also excludes unsafe permissions and escaping links.
        bundle.extractall(root, members=members, filter="data")
    return len(members)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-only", action="store_true",
                        help="Use the already downloaded archive")
    args = parser.parse_args()
    archive = DEST / "archive/source.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not args.extract_only:
        request = urllib.request.Request(URL, headers={"User-Agent": "PaperTranslation/0.1"})
        temporary = archive.with_suffix(".download")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if not response.geturl().startswith("https://arxiv.org/"):
                    raise ValueError("Download left the official arXiv host")
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    count = unpack(archive, DEST / "tex")
    print(f"Extracted {count} entries from {archive.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
