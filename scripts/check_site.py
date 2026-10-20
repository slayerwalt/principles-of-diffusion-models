"""Check built HTML links, fragments and local assets before Pages deployment."""

from __future__ import annotations

import argparse
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Document(HTMLParser):
    def __init__(self, path: Path):
        super().__init__(convert_charrefs=True)
        self.path = path
        self.anchors: set[str] = set()
        self.links: list[str] = []
        self.language = ""
        self.language_pairs: list[str] = []
        self.redirect = False
        self.unrendered_display_math = False
        self.feed(path.read_text(encoding="utf-8"))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "html":
            self.language = attrs.get("lang", "")
        if tag == "meta" and attrs.get("http-equiv", "").lower() == "refresh":
            self.redirect = True
        for key in ("id", "name" if tag == "a" else "id"):
            if attrs.get(key):
                self.anchors.add(attrs[key])
        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])
            if "data-language-pair" in attrs:
                self.language_pairs.append(attrs["href"])
        if tag in ("img", "script", "source") and attrs.get("src"):
            self.links.append(attrs["src"])
        if tag == "link" and attrs.get("href"):
            self.links.append(attrs["href"])

    def handle_data(self, data):
        # Arithmatex replaces display delimiters with math wrappers. A bare
        # delimiter means Markdown split the equation into ordinary text/code.
        if any(line.strip() == "$$" for line in data.splitlines()):
            self.unrendered_display_math = True


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
    original_links = [(path, href) for path, page in pages.items() for href in page.links
                      if urlsplit(href).netloc.endswith("arxiv.org")
                      and "2510.21890" in urlsplit(href).path]
    if len(original_links) != 1 or original_links[0][0] != root / "index.html":
        errors.append("The original paper should be linked exactly once, on the home page.")

    # Every public book page must have a matching language target. This also
    # catches a build that accidentally includes only the former preview UI.
    english_pages = [root / "index.html", *sorted((root / "en").rglob("*.html"))]
    for english in english_pages:
        if pages[english].redirect:
            continue
        chinese = root / "zh-CN" / (
            Path("index.html") if english == root / "index.html"
            else english.relative_to(root / "en")
        )
        for source, target, language in [
            (english, chinese, "en"), (chinese, english, "zh")
        ]:
            if source not in pages:
                errors.append(f"Missing bilingual book page: {source.relative_to(root)}")
                continue
            page = pages[source]
            if not page.language.startswith(language):
                errors.append(f"{source.relative_to(root)}: incorrect HTML language.")
            if len(page.language_pairs) != 1:
                errors.append(f"{source.relative_to(root)}: expected one language-switch target.")
                continue
            href = urlsplit(page.language_pairs[0])
            paired = (source.parent / unquote(href.path)).resolve()
            if paired.is_dir():
                paired /= "index.html"
            if paired != target:
                errors.append(f"{source.relative_to(root)}: incorrect language-switch target.")

    for internal in [
        "preview-index", "responsive-check.html", "language-menu-check.html",
        "en/samples", "zh-CN/samples", "translation-work", "reviews",
    ]:
        if (root / internal).exists():
            errors.append(f"Internal review material found in the public site: {internal}")
    search_index = root / "search.json"
    if not search_index.exists():
        errors.append("Search index is missing.")
    else:
        search = json.loads(search_index.read_text(encoding="utf-8"))
        if not {"en", "zh"}.issubset(search.get("config", {}).get("lang", [])):
            errors.append("Search index must enable English and Chinese tokenization.")

    for path, page in pages.items():
        if page.unrendered_display_math:
            errors.append(f"{path.relative_to(root)}: unrendered display math delimiter ($$).")
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
