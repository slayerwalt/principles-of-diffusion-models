"""Convert the pinned chapter 1 source into reviewable English Markdown.

This is a deliberately scoped converter, not a general LaTeX converter. It fails
on unhandled references and preserves the original words and mathematical content.
Requires Pandoc, Poppler's pdftoppm, and pypdf. No source code in the archive runs.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source/arxiv-2510.21890v3"
TEX = SOURCE / "tex"
CHAPTER = "Prologue-Deep-Generative-Modeling.tex"
OUTPUT = ROOT / "docs/en/chapters/01-deep-generative-modeling"
ASSETS = ROOT / "docs/en/assets/chapter-01"
PDF = ROOT / "The Principles of Diffusion Models.pdf"
PDF_URL = "https://arxiv.org/pdf/2510.21890v3"
FILES = ["index.md", "01-what-is-deep-generative-modeling.md", "02-examples.md",
         "03-taxonomy.md", "04-summary.md"]
LOCAL_REFS = {
    "ch:overview-dgm": ("Chapter 1", FILES[0], "ch-overview-dgm"),
    "sec:formulation-dgm": ("Section 1.1", FILES[1], "sec-formulation-dgm"),
    "subsec:intro-math-setup": ("Section 1.1.1", FILES[1], "subsec-intro-math-setup"),
    "sec:examples-dgm": ("Section 1.2", FILES[2], "sec-examples-dgm"),
    "sec:taxonomy": ("Section 1.3", FILES[3], "sec-taxonomy"),
    "sec:ch1_cr": ("Section 1.4", FILES[4], "sec-ch1-cr"),
    "fig:dgm-training": ("Figure 1.1", FILES[1], "fig-dgm-training"),
    "fig:ebm-ar-vae-nf-gan-dm-unified": ("Figure 1.2", FILES[2], "fig-ebm-ar-vae-nf-gan-dm-unified"),
    "tb:comparison-explicit-implicit": ("Table 1.1", FILES[3], "tb-comparison-explicit-implicit"),
    "eq:dgm-optimization": ("Equation (1.1.1)", FILES[1], "eq-dgm-optimization"),
    "eq:MLE": ("Equation (1.1.2)", FILES[1], "eq-mle"),
    "eq:fisher": ("Equation (1.1.3)", FILES[1], "eq-fisher"),
    "eq:f-div": ("Equation (1.1.4)", FILES[1], "eq-f-div"),
}
# PDF destinations, verified against the original document's link annotations.
EXTERNAL_REFS = {
    "ch:variational": ("Chapter 2", "chapter.2"),
    "ch:score-based": ("Chapter 3", "chapter.3"),
    "ch:score-sde": ("Chapter 4", "chapter.4"),
    "ch:flow-based": ("Chapter 5", "chapter.5"),
    "ch:ot-eot": ("Chapter 7", "chapter.7"),
    "eq:ebm-sm": ("Equation (3.1.3)", "equation.3.1.3"),
    "eq:sm": ("Equation (3.2.1)", "equation.3.2.1"),
    "eq: Monge's Formulation": ("Equation (7.2.2)", "equation.7.2.2"),
    "sec:vae": ("Section 2.1", "section.2.1"),
    "sec:ebm": ("Section 3.1", "section.3.1"),
    "sec:flow-based-method": ("Section 5.1", "section.5.1"),
}


def strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def argument(text: str, start: int) -> tuple[str, int]:
    while start < len(text) and text[start].isspace():
        start += 1
    if text[start] != "{":
        raise ValueError(f"Expected braced argument near {text[start:start + 70]!r}")
    depth = 1
    end = start + 1
    while depth:
        if text[end] in "{}" and (end == 0 or text[end - 1] != "\\"):
            depth += 1 if text[end] == "{" else -1
        end += 1
    return text[start + 1:end - 1], end


def replace_command(text: str, command: str, arity: int, render) -> str:
    pattern = re.compile(r"\\" + re.escape(command) + r"(?![A-Za-z])")
    pos = 0
    while match := pattern.search(text, pos):
        end = match.end()
        args = []
        for _ in range(arity):
            value, end = argument(text, end)
            args.append(value)
        result = render(*args)
        text = text[:match.start()] + result + text[end:]
        pos = match.start() + len(result)
    return text


def math_macros(text: str) -> str:
    replacements = {"bphi": r"{\boldsymbol{\phi}}", "btheta": r"{\boldsymbol{\theta}}",
                    "rvx": r"{\mathbf{x}}", "rvz": r"{\mathbf{z}}",
                    "rvf": r"{\mathbf{f}}", "rmG": r"{\mathbf{G}}",
                    "R": r"{\mathbb{R}}", "diff": r"\mathop{}\!\mathrm{d}",
                    "bm": r"\boldsymbol"}
    for name, value in replacements.items():
        text = re.sub(r"\\" + name + r"(?![A-Za-z])", lambda _: value, text)
    return replace_command(text, "abs", 1, lambda x: r"\left|" + x + r"\right|")


class Converter:
    def __init__(self, pandoc: str, reader: PdfReader):
        self.pandoc = pandoc
        self.reader = reader
        self.tokens: dict[str, str] = {}
        self.math_count = 0
        self.equation_count = 0
        self.figure_count = 0
        self.citation_keys: set[str] = set()
        self.external_targets: dict[str, dict] = {}
        result = subprocess.run([pandoc, str(TEX / "main_v2.bib"), "-f", "biblatex", "-t", "csljson"],
                                check=True, capture_output=True, encoding="utf-8")
        self.bibliography = {entry["id"]: entry for entry in json.loads(result.stdout)}

    def token(self, rendered: str) -> str:
        name = f"PDMPLACEHOLDER{len(self.tokens):05d}"
        self.tokens[name] = rendered
        return name

    def restore(self, text: str) -> str:
        # Some blocks contain earlier tokens (captions, links, or equations).
        while "PDMPLACEHOLDER" in text:
            previous = text
            for key, value in reversed(list(self.tokens.items())):
                text = text.replace(key, value)
            if previous == text:
                raise ValueError("Unknown conversion placeholder")
        return text

    def markdown(self, text: str) -> str:
        result = subprocess.run([self.pandoc, "-f", "latex", "-t",
                                 "markdown_strict+footnotes+tex_math_dollars+pipe_tables", "--wrap=none"],
                                input=text, check=True, capture_output=True, encoding="utf-8")
        if result.stderr.strip():
            raise ValueError(f"Pandoc reported a conversion warning: {result.stderr}")
        return result.stdout.strip()

    def reference(self, labels: str) -> str:
        links = []
        for label in labels.split(","):
            if label in LOCAL_REFS:
                title, filename, anchor = LOCAL_REFS[label]
                links.append(f"[{title}]({filename}#{anchor})")
            else:
                title, destination = EXTERNAL_REFS[label]
                page = self.reader.get_destination_page_number(self.reader.named_destinations[destination]) + 1
                self.external_targets[label] = {"title": title, "destination": destination, "pdf_page": page}
                links.append(f'[{title}]({PDF_URL}#page={page} "Original PDF; this chapter is not yet migrated")')
        return self.token(" and ".join(links))

    @staticmethod
    def author_year(entry: dict) -> str:
        names = [x.get("family", x.get("literal", "")) for x in entry["author"]]
        authors = names[0] if len(names) == 1 else " and ".join(names) if len(names) == 2 else names[0] + " et al."
        return f'{authors}, {entry["issued"]["date-parts"][0][0]}'

    def citations(self, labels: str) -> str:
        keys = labels.split(",")
        self.citation_keys.update(keys)
        links = [f'[{self.author_year(self.bibliography[k])}](references.md#ref-{k})' for k in keys]
        return self.token("(" + "; ".join(links) + ")")

    def display(self, match: re.Match) -> str:
        env, body = match.group("env"), match.group("body")
        self.math_count += 1
        labels = re.findall(r"\\label\{([^}]+)\}", body)
        body = re.sub(r"\\label\{[^}]+\}", "", body).strip()
        if env in ("align", "align*"):
            body = "\\begin{aligned}\n" + body + "\n\\end{aligned}"
        if env == "align":
            self.equation_count += 1
            body += f"\n\\tag{{1.1.{self.equation_count}}}"
        anchors = "\n".join(f'<span id="{LOCAL_REFS[label][2]}"></span>' for label in labels)
        result = (anchors + "\n\n" if anchors else "") + "$$\n" + math_macros(body) + "\n$$"
        return "\n\n" + self.token(result) + "\n\n"

    def prepare_math(self, text: str) -> str:
        pattern = r"\\begin\{(?P<env>align\*?|equation\*)\}(?P<body>.*?)\\end\{(?P=env)\}"
        text = re.sub(pattern, self.display, text, flags=re.S)
        def bracket(match):
            self.math_count += 1
            return "\n\n" + self.token("$$\n" + math_macros(match.group(1).strip()) + "\n$$") + "\n\n"
        text = re.sub(r"\\\[(.*?)\\\]", bracket, text, flags=re.S)
        text = re.sub(r"(?<!\\)\$(.*?)(?<!\\)\$", lambda m: self.token(
            "$" + math_macros(re.sub(r"\s+", " ", m.group(1)).strip()) + "$"), text, flags=re.S)
        return text

    def figure(self, match: re.Match) -> str:
        self.figure_count += 1
        block = match.group(0)
        caption = argument(block, block.index(r"\caption") + len(r"\caption"))[0]
        caption = replace_command(caption, "figcredit", 1, lambda credit: "\n\nSource: " + credit)
        caption = caption.replace(r"\textbfs", r"\textbf")
        caption = self.restore(self.markdown(self.prepare_math(caption)))
        label = re.search(r"\\label\{([^}]+)\}", block).group(1)
        asset = "dgm-learning.png" if self.figure_count == 1 else "computation-graphs.png"
        alt = "Target and model distributions with a dataset of samples" if self.figure_count == 1 else "Computation graphs for EBM, AR, VAE, NF, GAN, and diffusion models"
        output = (f'<span id="{LOCAL_REFS[label][2]}"></span>\n\n'
                  f'![{alt}](../../assets/chapter-01/{asset})\n\n'
                  f'**Figure 1.{self.figure_count}:** {caption}')
        return "\n\n" + self.token(output) + "\n\n"

    def bibliography_markdown(self) -> str:
        text = ["# Chapter 1 references {#chapter-01-references}",
                "These entries retain the metadata in the authors’ `main_v2.bib`. Links open the corresponding entries in the original PDF."]
        for key in sorted(self.citation_keys, key=lambda key: self.author_year(self.bibliography[key])):
            e = self.bibliography[key]
            authors = ", ".join(" ".join(filter(None, [a.get("given"), a.get("family")]))
                                if "family" in a else "et al." for a in e["author"])
            year = e["issued"]["date-parts"][0][0]
            venue = e.get("container-title", "")
            if e.get("volume"):
                venue += f' {e["volume"]}'
            if e.get("issue"):
                venue += f'({e["issue"]})'
            if e.get("page"):
                venue += f': {e["page"]}'
            if e.get("publisher"):
                venue += f'. {e["publisher"]}'
            dest = self.reader.named_destinations[f"cite.0@{key}"]
            page = self.reader.get_destination_page_number(dest) + 1
            text.extend([f'<span id="ref-{key}"></span>',
                         f'{authors} ({year}). “{e["title"]}.” {venue}. [Original bibliography]({PDF_URL}#page={page}).'])
        return "\n\n".join(text) + "\n"

    def convert(self, source: str) -> list[str]:
        text = strip_comments(source)
        text = replace_command(text, "epigraph", 2,
                               lambda quote, author: self.token("> " + self.markdown(quote) + "\n>\n> — " + author))
        text = replace_command(text, "chapterlearning", 2,
                               lambda items, route: r"\paragraph{After this chapter, you will be able to...}" +
                               r"\begin{itemize}" + items + r"\end{itemize}" +
                               r"\paragraph{Reader route.}" + route)
        text = replace_command(text, "lem", 3, lambda title, label, body:
                               "\n\n" + self.token(f'<span id="{label}"></span>\n\n**Lemma 1.1.1: {title}**') + "\n\n" + body)
        text = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", self.figure, text, flags=re.S)
        # Keep the TikZ figure with the model overview to which it belongs.
        figure_token = next(key for key, value in self.tokens.items() if "**Figure 1.2:**" in value)
        text = text.replace(figure_token, "")
        section_two = r"\section{Prominent Deep Generative Models}\label{sec:examples-dgm}"
        text = text.replace(section_two, section_two + "\n\n" + figure_token)
        table = ('<span id="tb-comparison-explicit-implicit"></span>\n\n'
                 '**Table 1.1: Comparison of Explicit and Implicit Generative Models**\n\n'
                 '| | Explicit: Exact Likelihood | Explicit: Approx. Likelihood | Implicit |\n'
                 '| --- | --- | --- | --- |\n'
                 '| **Likelihood** | Tractable | Bound/Approx. | Not Directly Modeled/Intractable |\n'
                 '| **Objective** | MLE | ELBO | Adversarial |\n'
                 '| **Examples** | NFs, ARs | VAEs, DMs | GANs |')
        text = re.sub(r"\\begin\{table\}.*?\\end\{table\}", lambda _: self.token(table), text, flags=re.S)
        text = replace_command(text, "Cref", 1, self.reference)
        text = replace_command(text, "citep", 1, self.citations)
        text = text.replace(r"\textbfs", r"\textbf").replace(r"\newpage", "")
        # Keep the two explicit roman labels referenced by the following prose.
        text = re.sub(r"\\begin\{enumerate\}(\s*\\item\[\(i\)\].*?)\\end\{enumerate\}",
                      lambda m: r"\begin{itemize}" + re.sub(r"\\item\[\((i+)\)\]",
                      lambda item: r"\item \textbf{(" + item.group(1) + ")} ", m.group(1)) +
                      r"\end{itemize}", text, flags=re.S)
        # Expand body math after figures (their captions have already been processed).
        text = self.prepare_math(text)
        text = replace_command(text, "paragraph", 1, lambda title: "\n\n" + r"\textbf{" + title + "} ")
        text = replace_command(text, "subparagraph", 1, lambda title: "\n\n" + r"\textbf{" + title + "} ")
        sections = re.split(r"\\section\{([^}]+)\}\\label\{([^}]+)\}", text)
        intro = replace_command(sections[0], "chapter", 1, lambda title: self.token(f"# 1. {title} {{#ch-overview-dgm}}"))
        intro = re.sub(r"\\label\{[^}]+\}", "", intro)
        outputs = [self.restore(self.markdown(intro))]
        for number in range(1, 5):
            title, label, body = sections[3 * number - 2:3 * number + 1]
            heading = f"# 1.{number} {title} {{#{LOCAL_REFS[label][2]}}}"
            subsection_number = 0
            def subsection(match):
                nonlocal subsection_number
                subsection_number += 1
                anchor = LOCAL_REFS[match.group(2)][2] if match.group(2) else "challenges-in-modeling-distributions"
                return "\n\n" + self.token(f"## 1.{number}.{subsection_number} {match.group(1)} {{#{anchor}}}") + "\n\n"
            body = re.sub(r"\\subsection\{([^}]+)\}(?:\\label\{([^}]+)\})?", subsection, body)
            outputs.append(heading + "\n\n" + self.restore(self.markdown(body)))
        # The lemma title contained inline math; keep it outside the bulk math scan.
        outputs = [math_macros(value) for value in outputs]
        if self.math_count != 31 or self.equation_count != 4 or self.figure_count != 2:
            raise ValueError(f"Unexpected chapter inventory: {self.math_count=} {self.equation_count=} {self.figure_count=}")
        if len(self.citation_keys) != 11:
            raise ValueError(f"Expected 11 references, got {len(self.citation_keys)}")
        return outputs


def render_assets(pdftoppm: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    subprocess.run([pdftoppm, "-singlefile", "-r", "180", "-png",
                    str(TEX / "Images/PartB/dgm-learning.pdf"), str(ASSETS / "dgm-learning")], check=True)
    # Original PDF physical page 34. Crop the plot only; retain the full caption as text.
    # At 180 dpi: x=58.8pt, y=64.8pt, width=390pt, height=466pt.
    subprocess.run([pdftoppm, "-f", "34", "-l", "34", "-singlefile", "-r", "180",
                    "-x", "147", "-y", "162", "-W", "975", "-H", "1165", "-png",
                    str(PDF), str(ASSETS / "computation-graphs")], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pandoc", default=shutil.which("pandoc"))
    parser.add_argument("--pdftoppm", default=shutil.which("pdftoppm"))
    parser.add_argument("--skip-assets", action="store_true")
    args = parser.parse_args()
    if not args.pandoc or (not args.pdftoppm and not args.skip_assets):
        parser.error("Pandoc and pdftoppm must be on PATH, or passed as explicit paths")
    converter = Converter(args.pandoc, PdfReader(PDF))
    outputs = converter.convert((TEX / CHAPTER).read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, body in zip(FILES, outputs):
        note = ('> English source conversion of arXiv:2510.21890v3, Chapter 1. '
                'References to chapters not yet migrated open the original PDF.\n\n')
        first, rest = body.split("\n", 1)
        (OUTPUT / filename).write_text(first + "\n\n" + note + rest.lstrip() + "\n", encoding="utf-8")
    (OUTPUT / "references.md").write_text(converter.bibliography_markdown(), encoding="utf-8")
    metadata = {"source": CHAPTER, "source_url": "https://arxiv.org/src/2510.21890v3",
                "pdf_physical_pages": [21, 34], "display_math_blocks": converter.math_count,
                "numbered_equations": converter.equation_count, "figures": converter.figure_count,
                "tables": 1, "footnotes": 3, "lemmas": 1,
                "citation_keys": sorted(converter.citation_keys),
                "external_references": converter.external_targets, "output_files": FILES + ["references.md"]}
    (SOURCE / "metadata").mkdir(exist_ok=True)
    (SOURCE / "metadata/chapter-01.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.skip_assets:
        render_assets(args.pdftoppm)
    print(f"Converted chapter 1: {len(outputs)} pages, 31 display equations, 2 figures, 11 references")


if __name__ == "__main__":
    main()
