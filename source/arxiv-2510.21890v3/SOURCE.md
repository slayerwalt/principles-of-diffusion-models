# 原始来源与转换记录

书名：The Principles of Diffusion Models。作者：Chieh-Hsin Lai、Yang Song、Dongjun Kim、Yuki Mitsufuji、Stefano Ermon。

本次固定使用 arXiv `2510.21890v3`，不自动追踪最新版。官方页面为 <https://arxiv.org/abs/2510.21890v3>，源码为 <https://arxiv.org/src/2510.21890v3>，PDF 为 <https://arxiv.org/pdf/2510.21890v3>。源码于 2026-10-19 下载。

原著采用 [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) 许可。原著及图像的版权属于作者；本项目保留署名和来源，英文网页为格式转换。本地迁移不代表作者背书。中文翻译的公开分发需要另行取得权利人的授权。

`archive/source.tar.gz` 是官方原始压缩包，`tex/` 是未经修改的解包文件。解包器检查路径穿越、绝对路径、符号链接和特殊文件，仅接受常规文件和目录；本项目不执行源码包中的程序或脚本。

## 复现命令

以下命令从项目根目录执行，需要 Python 3.12 或更新版本、Pandoc 和 Poppler 的 `pdftoppm`；转换器还需要 Python 包 `pypdf`。首次转换使用 Pandoc 3.6.1。网站构建直接读取已提交的 Markdown 和图片，无需安装这些转换依赖。

```powershell
python scripts/fetch_arxiv_source.py
python scripts/convert_chapter_01.py
```

如果压缩包已经下载，可以只执行安全解包：

```powershell
python scripts/fetch_arxiv_source.py --extract-only
```

`convert_chapter_01.py` 还依赖项目根目录中现有的 `The Principles of Diffusion Models.pdf`，用于读取跨章引用的真实目标页和提取内嵌 TikZ 图。工具不在 PATH 时，可传入 `--pandoc` 和 `--pdftoppm` 的绝对路径。`--skip-assets` 只更新文字和引用。

## 样章范围与处理方式

英文样章覆盖第 1 章 Deep Generative Modeling 全文，对应 PDF 物理页 21–34（书内页 16–29），源文件为 `tex/Prologue-Deep-Generative-Modeling.tex`。输出是章引言、1.1–1.4 四个节页面，以及本章引用的 11 条文献。章引言含题词、学习目标和 Reader route。

转换先剥除 LaTeX 注释，再处理作者自定义宏、数学环境、交叉引用、图表，最后由 Pandoc 转换普通文字、强调、列表和脚注。正文保留原英文措辞，包括原文中可能需要作者进一步澄清的表述。转换器专用于这一章，遇到未知引用或数量不符会报错，不能直接用于后续所有章节。

数学内容保留为可编辑 TeX，展开 `\rvx`、`\bphi`、`\bm`、`\diff` 等宏。31 个显示数学块中，4 个原有编号保留为 `(1.1.1)` 至 `(1.1.4)`，其余保持无编号；Lemma 1.1.1 保留原编号。数学符号及推导未作内容改写。

图 1.1 从作者的 `Images/PartB/dgm-learning.pdf` 渲染为 PNG。图 1.2 在源码中为 TikZ，因此从现有官方 PDF 物理第 34 页裁取图形区域，图注和作者署名仍以可编辑正文保留。图 1.2 在网页中安排于第 1.2 节开头；PDF 的浮动图排于本章末尾。两幅图均保留完整图形，未重新绘制。

表 1.1 的表头改为网页支持的单层列标题，保留 Explicit / Implicit 和 Exact / Approx. Likelihood 的全部层级含义及各单元格内容。概率密度的两项要求保留 `(i)`、`(ii)` 标记，供后续正文引用。脚注通过网页脚注链接跳转。

章内链接指向稳定的锚点。尚未迁移章节的引用指向官方 PDF 的对应物理页，并在页面说明及链接提示中标明。编号和页码取自 PDF 的真实内部链接目标，记录于 `metadata/chapter-01.json`。本章文献从作者的 `main_v2.bib` 提取，文献链接指向官方 PDF 对应文献页。

重新运行转换会覆盖本章的生成 Markdown。后续逐章校对发现问题时，应同步修正转换脚本或建立明确的校对补丁，避免手工修订在重跑时丢失。

## 核验清单

- 有效正文包括 4 节、2 个小节、章引言、题词、学习目标、Reader route 和全部结束语。
- 显示数学 31 块，编号公式 4 个，引理 1 个；没有把注释中的旧推导混入正文。
- 图 2 幅，表 1 张，脚注 3 个；两幅输出图已逐一检查边界和可读性。
- 8 处文献引用涵盖 11 个文献键；16 处交叉引用包含一处双目标引用。
- 本章所有显式标签及引理锚点均保留；未迁移内容的链接使用官方 PDF。
- 网站严格构建通过，生成的 11 个 HTML 页面的本地链接、锚点及资源检查通过。
- 浏览器检查通过：各节数学没有 MathJax 错误，图片均加载成功；409px 窄屏页面没有整体横向溢出，1265px 桌面布局可显示章节目录和页内目录；搜索能定位 Fisher divergence，明暗切换可用。
