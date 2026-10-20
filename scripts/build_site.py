"""Build the bilingual book from its public source files."""

from pathlib import Path
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    # Maintain bibliography entries only in the English source.
    original = (ROOT / "docs/en/references.md").read_text(encoding="utf-8")
    references = original.replace(
        "# References {#references}", "# 参考文献 {#references}", 1
    ).replace(
        "References cited in the preface, chapters, and appendices are collected "
        "here, in alphabetical order by the first author’s surname.",
        "本页汇集前言、各章和附录引用的参考文献，按第一作者姓氏的字母顺序排列。",
        1,
    )
    (ROOT / "docs/zh-CN/references.md").write_text(references, encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "zensical", "build", "--clean", "--strict"],
        cwd=ROOT,
        check=True,
    )

    # Zensical 0.0.71 emits one index. Its bundled worker supports Chinese
    # segmentation through Intl.Segmenter when "zh" is in the language list.
    index_path = ROOT / "site/search.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["config"]["lang"] = ["en", "zh"]
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
