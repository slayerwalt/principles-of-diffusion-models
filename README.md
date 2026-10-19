# The Principles of Diffusion Models 网页版

本项目将《The Principles of Diffusion Models》的英文原文按章节整理为可维护的网页，基于 arXiv `2510.21890v3`（2026-08-27）。先完成英文转换与校对，再进行中文翻译。

仓库：<https://github.com/slayerwalt/principles-of-diffusion-models>

在线阅读：<https://slayerwalt.github.io/principles-of-diffusion-models/>

[开始阅读第 1 章英文样章](https://slayerwalt.github.io/principles-of-diffusion-models/en/chapters/01-deep-generative-modeling/)。后续推送到 `main` 的更新会通过 GitHub Actions 自动发布。

## 当前内容

网站包含完整书目导航、第 1 章英文样章、来源与许可说明，以及中文翻译计划。第 1 章分为章首页和 1.1–1.4 四个小节，保留原文公式、编号、图片、表格、脚注和参考文献。其他章节在目录中标为尚未转换。

网站使用 Zensical，具有章节导航、页内目录、搜索、明暗模式和移动端布局。数学公式通过随仓库保存的 MathJax 3.2.2 渲染，字体使用系统字体。Zensical 是 Material for MkDocs 团队维护的后继项目；选型依据见其[官方介绍](https://zensical.org/docs/get-started/)及 Material 的[维护期限说明](https://squidfunk.github.io/mkdocs-material/getting-started/)。

## 本地预览

需要 Python 3.12。在项目目录中执行以下 PowerShell 命令；已经配置好 `.venv` 时可以直接运行最后一行。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m zensical serve --dev-addr 127.0.0.1:8000
```

访问 <http://127.0.0.1:8000/principles-of-diffusion-models/>。结束预览时在终端按 `Ctrl+C`。

macOS / Linux 可将上述 Python 路径替换为 `.venv/bin/python`。

构建和校验：

```powershell
.\.venv\Scripts\python.exe -m zensical build --clean --strict
.\.venv\Scripts\python.exe scripts/check_site.py
```

构建结果位于 `site/`。校验脚本检查生成网页中的本地页面、锚点、图片、脚本和样式资源；公式的实际排版仍需浏览器检查。

## 目录约定

```text
docs/
  index.md                         站点首页
  en/
    book-outline.md                全书目录与迁移状态
    about.md                       原文来源、作者与许可
    chapters/
      01-deep-generative-modeling/  第 1 章，按节分文件
    assets/chapter-01/              样章图片
  zh-CN/index.md                   中文翻译计划
  assets/                          网站样式、数学配置和 MathJax
source/arxiv-2510.21890v3/           来源记录与本地源码缓存
scripts/                           下载、转换和构建检查脚本
zensical.toml                      网站与导航配置
.github/workflows/pages.yml         GitHub Pages 构建和发布流程
```

源码压缩包、完整解包目录、原 PDF、虚拟环境和构建目录都通过 `.gitignore` 排除。已转换的 Markdown、图像、来源记录以及浏览器运行所需的数学资源会进入仓库；日常构建无需重新下载论文或安装 LaTeX。

## 重新获取源码与转换样章

`source/arxiv-2510.21890v3/` 保存具体来源及转换记录。官方源码下载地址固定为 <https://arxiv.org/src/2510.21890v3>，下载脚本会检查解包路径，并且不执行归档中的代码。

```powershell
.\.venv\Scripts\python.exe scripts/fetch_arxiv_source.py
.\.venv\Scripts\python.exe -m pip install -r requirements-convert.txt
.\.venv\Scripts\python.exe scripts/convert_chapter_01.py
```

重新转换还需要 Pandoc、Poppler 的 `pdftoppm` 位于 `PATH`，以及根目录中的 `The Principles of Diffusion Models.pdf`（同一 v3 版本）。原 PDF 用于取得跨章引用目标，并提取源文件中内嵌绘制的图 1.2；图 1.1 从源码包中的独立 PDF 图像生成。可以通过 `--pandoc`、`--pdftoppm` 参数指定程序路径。

转换脚本专门适配第 1 章。它会重写该章的生成文件，因此人工修正应同步进入转换逻辑；扩展到其他章节前，需要处理相应的宏、定理环境和引用映射。未迁移章节的引用链接到原文 PDF，待该章转换后再换成本地锚点。

MathJax 已随仓库保存。仅在需要恢复该固定版本时运行：

```powershell
.\.venv\Scripts\python.exe scripts/vendor_mathjax.py
```

## 发布到 GitHub Pages

本地准备完成后，将文件提交并推送到 `main`。在 GitHub 仓库中打开 **Settings → Pages → Build and deployment**，将 **Source** 设置为 **GitHub Actions**。首次设置后可在 Actions 中手动运行 **Build and publish book**。

工作流会安装锁定依赖、严格构建、检查内部链接，再发布 `site/`。PR 只执行构建检查；推送 `main` 或在 `main` 手动运行时才部署。仓库使用默认的 `GITHUB_TOKEN`，无需配置个人访问令牌。

## 版本与翻译

英文基线固定为 v3。新增章节时继续使用稳定的英文文件名和原文编号，并更新 `zensical.toml` 与全书目录。英文版完成后，在 `docs/zh-CN/` 按相同章节结构翻译，同时维护术语表及每章对应的英文提交版本。

## 署名与许可

原作者为 Chieh-Hsin Lai、Yang Song、Dongjun Kim、Yuki Mitsufuji、Stefano Ermon。原文采用 [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)，本项目是非官方、非商业的格式转换。原文权利保留在作者及相关权利人处；整个仓库不能统一视为 MIT 等宽松许可。

中文译本的公开传播需要另行取得相应授权。MathJax 的 Apache-2.0 许可保留在其资源目录内。网页中的原文内容以[官方 PDF](https://arxiv.org/pdf/2510.21890v3)为校对依据。
