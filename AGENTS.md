# 项目维护说明

本文件记录截至 2026-10-19 的项目情况，适用于整个仓库。后续完成章节迁移或调整维护流程时，同步更新相关记录。用户在当前会话中的最新要求优先于本文件。

## 项目目标与当前状态

本项目将《The Principles of Diffusion Models》的英文原文按章节整理为 GitHub Pages 网站。先完成整本英文版的转换与校对，再开展中文翻译。当前固定使用 arXiv `2510.21890v3`（2026-08-27），不要自动切换到其他版本。

仓库为 <https://github.com/slayerwalt/principles-of-diffusion-models>，网站为 <https://slayerwalt.github.io/principles-of-diffusion-models/>。本地工作目录是 `D:\project\PaperTranslation`；下文中的相对路径均以项目根目录为基准。

原 PDF 共 522 页，包含 12 章正文和 4 个附录。第 1 章 Deep Generative Modeling 已完整转换，覆盖 PDF 物理页 21–34、书内页 16–29，包含章引言、1.1–1.4 四节和本章参考文献。已核对保留 31 块显示公式、4 个编号公式、1 个引理、2 幅图、1 张表、3 条脚注和 11 条文献；31 块显示公式已与源码逐项比对。

第 2 章 Variational Perspective: From VAEs to DDPMs 已完整转换，覆盖 PDF 物理页 42–67、书内页 37–62，包含章引言、2.1–2.3 三节和本章参考文献。原文清单为 68 块显示公式、18 个编号公式、3 个定理及其证明段落、1 个引理、1 个编号问题、1 个编号 Observation、1 个 Remark、7 幅图、5 条脚注和 5 条文献；本章无表格和算法。68 块显示公式和 242 处行内公式已与源码独立核对，浏览器确认 VAE 页 29 块、DDPM 页 39 块显示公式全部渲染。

第 3、4、5、6、7、9 章、Part II 无编号导读和附录 D 已有供正文引用的占位入口，内容标为 `Not yet available`，尚未完成转换。其他未迁移内容在全书目录中说明。`docs/zh-CN/index.md` 仅记录翻译计划，目前没有中文译文章节。

网站使用 Zensical 0.0.71 和 MathJax 3.2.2，支持章节导航、搜索、明暗模式及窄屏阅读。MathJax 在 GitHub Actions 构建时下载，并与静态网页一起打包发布。加入第二章后，构建生成 24 个 HTML 页面；此数量是当时的记录，不应作为新增章节后的固定断言。每次内容更新都要重新完成相应核对，并检查当次线上部署结果。

## 用户约束

- 对话、计划、审核材料和维护说明默认使用中文；英文原文转换阶段保留英文正文。
- **之后所有新 Git 提交信息使用英文。** 已有中文提交保留，不因这条要求重写历史。提交消息不添加 AI 的 `Co-authored-by`；只有真人协作者可使用该标记。
- 默认不使用 Git worktree，除非用户特别要求。
- 不主动计算文件的 SHA-256 等内容指纹，除非用户要求。
- **在线内容只在首页开头说明原论文地址。** 正文、参考文献列表、其他介绍页和章节占位页都不再链接到本书的原 PDF 或 arXiv 页面；需要介绍来源时链接到站内首页的 `#source`。
- 跨章、节、公式引用使用站内路径和稳定锚点。尚未迁移的目标要建立明确的占位入口，不能用原论文外链替代，也不能伪造正文。
- README 只保留项目概况、阅读入口、内容进度和版权说明。不要重新加入开发教程、本地预览、目录约定或构建部署说明。
- 使用单个 `main` 分支维护书稿及最小在线构建输入，通过 GitHub Actions 构建并直接发布 Pages 产物。用 `.gitignore` 排除原始材料、转换工具、第三方下载目录和本地产物；不要恢复 `gh-pages` 发布分支。当前不使用 GitBook.com，也不使用其他托管平台替代 GitHub Pages。
- 表达以清晰自然段为主，列表用于并列或顺序信息。避免无意义的标题、装饰性 Markdown、空泛套话和重复总结。
- 验证范围应与改动相符。不要为可撤销的小改动编写照搬实现的测试；检查通过后，除非出现新改动、失败或未解决的问题，不重复扩大测试范围。

## 文件、分支与本地环境

`main` 保存书稿 Markdown、原书插图、简要 README、`.gitignore`、本维护说明及最小在线构建输入。章节位于 `docs/en/chapters/`，每章使用一个目录，章引言为 `index.md`，各节独立成页。`docs/en/book-outline.md` 维护全书目录和转换状态。

以下在线构建输入需要跟踪并保留在 `main`：

- `zensical.toml`、`requirements.txt`：站点配置和固定版本的网站构建依赖。
- `.github/workflows/pages.yml`：GitHub Actions 构建、检查和 Pages 发布流程。
- `docs/assets/stylesheets/book.css`、`docs/assets/javascripts/mathjax.js`：网站样式和数学渲染配置。
- `scripts/check_site.py`、`scripts/vendor_mathjax.py`：生成网站的完整性检查和固定版本 MathJax 下载脚本。

`docs/en/assets/` 保存需要跟踪的原书插图。`docs/assets/` 中的上述样式和配置文件也需要跟踪；只有第三方下载目录 `docs/assets/vendor/` 继续被忽略。CI 运行 `scripts/vendor_mathjax.py` 下载 MathJax 3.2.2，再将这些资源及其许可打包进网站。清理 Git 跟踪时，不要误删插图或在线构建输入。

以下内容由 `.gitignore` 排除，在需要时保留于本机：

- `requirements-convert.txt`、`scripts/fetch_arxiv_source.py`、`scripts/convert_chapter_01.py` 等转换依赖和工具。
- `source/` 和根目录原 PDF：原始 PDF、LaTeX 源码及本地转换记录。
- `docs/assets/vendor/`：MathJax 等第三方下载资源。
- `.venv/`、`.cache/`、`site/`、`tmp/`、`output/`、`.pages-publish/`、`LOCAL_MAINTENANCE.md`：本地环境、产物和维护缓存。
- 旧的本地发布脚本 `scripts/publish_pages.py`：已停止使用，应保持禁用，避免重新建立或推送 `gh-pages`。

克隆 `main` 后，GitHub Actions 可通过跟踪的依赖、配置和脚本完成网站构建。正常发布由远端工作流完成，用户无需先在本机构建。章节重新转换另需恢复本地原始材料和转换工具。

当前 `.venv` 为 Python 3.12.14，已安装网站构建依赖。重新运行转换器前须安装 `requirements-convert.txt` 中的 `pypdf`；截至本次记录，该包尚未安装到项目 `.venv`。不要假定系统默认的 `python` 就是本项目解释器。

Pandoc 和 Poppler 的 `pdftoppm` 已在当前环境的 PATH 中。工具不可用时，检查实际安装位置，或使用转换器的 `--pandoc`、`--pdftoppm` 参数指定路径。

## 原始材料与转换方法

官方源码已保存在 `source/arxiv-2510.21890v3/archive/source.tar.gz`，未经修改的解包目录为 `source/arxiv-2510.21890v3/tex/`。主文件是 `main_v2.tex`，参考文献是 `main_v2.bib`；第 1 章源文件是 `Prologue-Deep-Generative-Modeling.tex`，第 2 章源文件是 `Chapter-1-Varational-Perspective-From-VAE-to-DDPM.tex`。文件名中的章号与成书章号不同，以 PDF 和主文件顺序为准。根目录的 `The Principles of Diffusion Models.pdf` 是同一 v3 版本的排版核对依据。

转换优先从 LaTeX 获取结构、正文、公式和文献，再对照 PDF 检查。不要以 PDF 纯文字提取代替数学源码，也不要把完整章节改写为摘要。

1. 确认章节在主文件中的位置、相关宏定义和 PDF 页码，清点标题、数学块、原编号、图表、脚注、文献及交叉引用。每章建立自己的清单，不能照搬第 1 章的数量。
2. 剥除 LaTeX 注释和被注释的旧稿、作者讨论。读取源码内容，不执行归档中的程序或脚本；下载解包时检查路径穿越、链接和特殊文件。
3. 展开自定义宏为 MathJax 可处理的 TeX，例如 `\rvx`、`\bphi`、`\bm`、`\diff`。行内公式用 `$...$`，显示公式用 `$$...$$`；需要时把 `align` 包装为 `aligned`，保留原公式编号及 `\tag{...}`。
4. 为章节、节、公式、图、表和引理建立显式锚点，并把 LaTeX 标签引用转换为 Markdown 站内链接。显示编号与链接锚点分别保留，避免标题修改导致引用失效。
5. 普通文字、强调、列表和脚注通过 Pandoc 转换，再逐段核对。保留原英文措辞、数学含义和推导；疑似原文问题单独记录，不静默修改作者结论。
6. 图像优先使用作者提供的原始资源。第 1 章图 1.1 从 `Images/PartB/dgm-learning.pdf` 渲染为 PNG；图 1.2 为内嵌 TikZ，从原 PDF 物理第 34 页裁取完整图形。图注、编号和署名保留为可编辑正文，裁图后检查边界和可读性。
7. 表格可为网页重排，但保留表头层级含义和全部单元格内容。文献从 `.bib` 提取，保留作者、年份、标题、出处以及 `ref-...` 锚点；不要再附加指向本书原 PDF 文献页的链接。
8. 按章、节输出 Markdown，更新全书目录与 `zensical.toml` 导航，完成内容核对后再构建和发布。

`scripts/convert_chapter_01.py` 是**仅适配第 1 章**的转换器。它会重写该章的 6 个 Markdown 文件；默认同时生成图片，`--skip-assets` 仅跳过图片生成，仍需要根目录原 PDF。重跑前检查人工修订，并把必要修正同步到转换逻辑或明确的补丁中。未知引用、未知宏和数量异常应被检查处理，不能静默丢弃。

`scripts/convert_chapter_02.py` 仅适配第 2 章，输出章引言、VAE、DDPM、Closing Remarks 和参考文献共 5 个 Markdown 文件。图片由 `scripts/extract_chapter_02_figures.py` 提取到 `docs/en/assets/chapter-02/`：3 幅 TikZ 图和图 2.5 从原书 PDF 裁取，其余 3 幅图从作者提供的 PDF 资源渲染。图 2.5 的独立资源单独渲染时丢失一处标签，因此使用原书页面中的完整图形。转换清单保存在本地 `source/arxiv-2510.21890v3/metadata/chapter-02.json`，独立核对记录在 `tmp/chapter02-audit/`。这些脚本和记录继续由 `.gitignore` 忽略。

当前转换器使用 `LOCAL_REFS` 和 `PENDING_REFS`，重跑不会恢复原 PDF 外链。以下已发布路径和锚点需要持续兼容：

- `02-variational-perspective/index.md`：`ch-variational`、旧 `sec-vae` 入口；2.1 节正文位于 `02-variational-perspective/01-variational-autoencoder.md#sec-vae`，已有正文链接已更新。
- `03-score-based-perspective/index.md`：`ch-score-based`、`sec-ebm`、`eq-ebm-sm`、`sec-ebm-to-score`、`eq-sm`。
- `04-score-sde-framework/index.md`：`ch-score-sde`、`dsm-minimizer`（命题 4.3.1）、`sec-variational-perspective`（4.5 节）、`reverse-kernel`（引理 4.5.2）。
- `05-flow-based-perspective/index.md`：`ch-flow-based`、`sec-flow-based-method`。
- `07-optimal-transport/index.md`：`ch-ot-eot`、`eq-monge-formulation`。

第二章新增的跨章入口还包括第 3 章 `sec-smld`（3.4 节）、第 6 章 `06-unified-perspective/index.md#equi-para`（命题 6.3.1）及第 9 章 `09-numerical-solvers/index.md#sec-revisit-ddim`（9.2 节）。附录证明入口为 `docs/en/appendices/d-supplementary-proofs/index.md#subsec-proof-equiv-marginal-kl`（D.1.1）和 `#app-vlb-proof`（D.1.2）。

第二章引用的公式 (1.4.1) 实际位于 Part II 的无编号导读，使用 `docs/en/foundations/operational-view.md#eq-ddpm-first-try` 占位；保留原编号，不要误连第 1 章的 Closing Remarks。

上述路径均相对 `docs/en/chapters/`。填充这些章节时，应保留现有入口；若把内容拆到其他页面，需同时处理已有链接和旧入口的兼容。

重新转换第 1 章的命令：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-convert.txt
.\.venv\Scripts\python.exe scripts/convert_chapter_01.py
```

已有源码无需重复下载。确需恢复时使用 `scripts/fetch_arxiv_source.py`；`--extract-only` 可从现有压缩包重新安全解包。记录性文档编辑不需要运行转换器。

`source/arxiv-2510.21890v3/SOURCE.md` 是首次转换的历史记录，其中“引用跳转原 PDF”和“11 个 HTML 页面”等描述已过时。不要用这些旧描述覆盖当前站内引用规则和发布流程。

## 验证与发布

转换后逐段核对正文、公式及原编号，检查图注、脚注、文献和引用目标。对复杂公式、图片、表格以及布局改动进行浏览器检查，确认没有 MathJax 错误、图片缺失或整页横向溢出。只修改维护说明时，无需重建网站。

GitHub Actions 负责安装 `requirements.txt` 中的网站依赖、运行 `scripts/vendor_mathjax.py` 下载固定版本资源、严格构建网站，并运行 `scripts/check_site.py`。需要在本地排查内容或布局问题时，可以执行相同的基本检查：

```powershell
.\.venv\Scripts\python.exe scripts/vendor_mathjax.py
.\.venv\Scripts\python.exe -m zensical build --clean --strict
.\.venv\Scripts\python.exe scripts/check_site.py
```

检查脚本验证本地页面、锚点、图片、脚本、样式及 MathJax 字体，检测 HTML 中未转换的显示公式分隔符，并要求本书原文链接在首页恰好出现一次。显示公式内部不能保留 Markdown 空白段落，否则公式可能被拆成普通文字或代码块；浏览器中还要比对实际渲染数量与源码清单。检查脚本不能替代正文和数学内容的人工核对。

Pages 的发布来源设置为 **GitHub Actions**，保留 HTTPS。`.github/workflows/pages.yml` 在远端构建和检查 `main` 的内容，通过后直接上传 Pages 产物并部署。生成的静态网页不提交到 `main`，也不依赖 `gh-pages` 分支。

需要发布网站更新时，先按当次任务范围确认修改，用英文消息提交并推送 `main`，再检查对应 Actions 运行和 Pages 部署结果。无需运行旧的 `scripts/publish_pages.py`。`.pages-publish/` 仅为遗留本地缓存，继续忽略，不再参与发布。

完成发布后，确认 GitHub Actions 的构建、检查及 Pages 部署成功，再验证受影响的线上页面；不能仅凭推送成功或本地构建通过就宣称网站更新完成。

## 版权与署名

原作者为 Chieh-Hsin Lai、Yang Song、Dongjun Kim、Yuki Mitsufuji 和 Stefano Ermon。原作采用 CC BY-NC-ND 4.0，正文、公式、图表和第三方材料需保留相应署名及许可。本项目为非官方、非商业的格式整理，不代表作者背书，不要给整个书稿仓库套用 MIT 等宽松许可证。

中文译本的公开传播需要另行取得相关权利人的授权。MathJax 的 Apache-2.0 许可保留在网站运行资源中；部署静态网页时一并保留。
