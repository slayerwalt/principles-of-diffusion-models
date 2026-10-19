"""Download the pinned MathJax browser distribution for self-hosted equations."""

from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request

VERSION = "3.2.2"
URL = f"https://registry.npmjs.org/mathjax/-/mathjax-{VERSION}.tgz"
ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "tmp" / f"mathjax-{VERSION}.tgz"
DEST = ROOT / "docs" / "assets" / "vendor" / "mathjax"


def main():
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.is_file():
        request = urllib.request.Request(URL, headers={"User-Agent": "DiffusionBookMigration/1.0"})
        temporary = ARCHIVE.with_suffix(".download")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
            temporary.replace(ARCHIVE)
        finally:
            temporary.unlink(missing_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if not member.isfile():
                continue
            if path.parts[:2] == ("package", "es5"):
                relative = PurePosixPath(*path.parts[2:])
            elif member.name == "package/LICENSE":
                relative = PurePosixPath("LICENSE")
            else:
                continue
            target = (DEST / str(relative)).resolve()
            if not target.is_relative_to(DEST.resolve()):
                raise ValueError(f"Unsafe archive member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            count += 1
    if not (DEST / "tex-mml-chtml.js").is_file() or not (DEST / "LICENSE").is_file():
        raise RuntimeError("Incomplete MathJax browser distribution")
    (DEST / "NOTICE.txt").write_text(
        f"MathJax {VERSION}\nSource: {URL}\nLicense: Apache-2.0; see LICENSE.\n"
        "Browser distribution copied without modification.\n", encoding="utf-8"
    )
    print(f"Vendored MathJax {VERSION}: {count} files in {DEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
