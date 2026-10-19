"""Check built HTML links, fragments and local assets before Pages deployment."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Document(HTMLParser):
    def __init__(self, path: Path):
        super().__init__(convert_charrefs=True)
        self.path = path
        self.anchors: set[str] = set()
        self.links: list[str] = []
        self.feed(path.read_text(encoding="utf-8"))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for key in ("id", "name" if tag == "a" else "id"):
            if attrs.get(key):
                self.anchors.add(attrs[key])
        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])
        if tag in ("img", "script", "source") and attrs.get("src"):
            self.links.append(attrs["src"])
        if tag == "link" and attrs.get("href"):
            self.links.append(attrs["href"])


def check(root: Path, base_url: str) -> list[str]:
    root = root.resolve()
    base = urlsplit(base_url)
    prefix = base.path.rstrip("/")
    pages = {p.resolve(): Document(p) for p in root.rglob("*.html")}
    errors: list[str] = []
    if root / "index.html" not in pages:
        return ["site/index.html is missing; build the website first."]
    # MathJax loads these at runtime, so HTML href/src checks cannot see them.
    fonts = root / "assets/vendor/mathjax/output/chtml/fonts/woff-v2"
    if not (fonts / "MathJax_Main-Regular.woff").is_file():
        errors.append("MathJax runtime fonts are missing from the built site.")

    for path, page in pages.items():
        for href in page.links:
            url = urlsplit(href)
            if url.scheme and url.scheme not in ("http", "https"):
                continue
            if url.netloc and url.netloc != base.netloc:
                continue
            raw_path = unquote(url.path)
            if raw_path.startswith("/"):
                if prefix and (raw_path == prefix or raw_path.startswith(prefix + "/")):
                    raw_path = raw_path[len(prefix):]
                target = root / raw_path.lstrip("/")
            else:
                target = path.parent / raw_path if raw_path else path
            target = target.resolve()
            if not target.is_relative_to(root):
                errors.append(f"{path.relative_to(root)}: URL leaves site directory: {href}")
                continue
            if target.is_dir():
                target /= "index.html"
            if not target.is_file():
                errors.append(f"{path.relative_to(root)}: missing target: {href}")
            elif url.fragment and target in pages:
                fragment = unquote(url.fragment)
                if fragment not in pages[target].anchors:
                    errors.append(f"{path.relative_to(root)}: missing anchor: {href}")

    if not errors:
        print(f"Checked {len(pages)} HTML pages: internal links, anchors and local assets OK.")
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    parser.add_argument("--base-url", default="https://slayerwalt.github.io/principles-of-diffusion-models/")
    args = parser.parse_args()
    failures = check(args.site_dir, args.base_url)
    for failure in failures:
        print(failure)
    raise SystemExit(bool(failures))
