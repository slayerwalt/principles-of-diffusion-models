# 前言与阅读路线 {#preface}

扩散模型已迅速成为生成建模的核心范式，相关研究数量庞大，涵盖机器学习、计算机视觉、自然语言处理等众多领域。这些文献分散在不同研究社区中，关注不同方面的进展：理论基础涉及建模原理、训练目标、采样器设计及其背后的数学思想；实现方面的进展涵盖工程实践与架构选择；实际应用将模型适配到特定领域或任务；系统层面的优化则提升计算、内存使用和部署的效率。

本书旨在建立扩散模型的*原理基础*，重点围绕以下主题展开：

-   我们介绍支撑扩散模型研究的基本概念与表述，使读者获得阅读更广泛文献所需的核心理解。我们不全面综述所有变体或特定领域的应用，而是建立稳定的概念基础，帮助读者理解这些发展。

-   经典生成模型学习从噪声到数据的直接映射，而扩散模型将生成视为随时间逐渐变换的过程，从粗略结构逐步细化出精细细节。这一核心思想主要通过三种视角发展起来，即*变分*、*基于 score* 和*基于流*的方法，它们为理解和实现扩散建模提供了互补的方式。我们聚焦这些表述的核心原理与基础，旨在追溯其中关键思想的起源，阐明不同表述之间的关系，并形成连贯的理解，将直觉认识与严格的数学表述联系起来。

-   在这些基础之上，我们考察如何进一步发展扩散模型，以更高效地生成样本、更好地控制生成过程，并启发建立在扩散原理之上的独立生成建模形式。

本书面向对深度学习具有基本了解（例如，知道什么是神经网络以及如何训练），或更具体地说，对深度生成建模具有基本了解，并希望超越表面认识、加深对扩散模型理解的研究者、研究生和从业者。读完本书后，读者将从原理层面理解扩散建模的基础，能够在统一框架内解读不同表述，并具备熟练应用现有模型和探索新研究方向所需的背景知识。

## 本书阅读路线 {#roadmap-of-this-book}

本书系统介绍扩散模型的基础，并追溯其背后的核心原理。

### 建议的阅读顺序 {#suggested-reading-path}

我们建议按照本书的编排顺序阅读，以建立全面的理解。已熟悉基础知识的读者可以跳过标有*选读*的小节。例如，熟悉深度生成模型（DGM）的读者可以跳过[第 1 章](../chapters/01-deep-generative-modeling/index.md#ch-overview-dgm)的概述。同样，已经了解变分自编码器（[第 2.1 节](../chapters/02-variational-perspective/01-variational-autoencoder.md#sec-vae)）、基于能量的模型（[第 3.1 节](../chapters/03-score-based-perspective/01-energy-based-models.md#sec-ebm)）或归一化流（[第 5.1 节](../chapters/05-flow-based-perspective/01-normalizing-flows-and-neural-odes.md#sec-flow-based-method)）的读者，也可以跳过这些入门小节。其他选读部分进一步介绍高级或专门主题，可根据需要查阅。

本书分为四个主要部分。

### A、B 两部分：扩散模型的基础 {#parts-a-b-foundations-of-diffusion-models}

本部分回顾塑造该领域的三种基础视角，追溯扩散模型的起源[^1]。[图 3](preface.md#fig-part-b-roadmap)给出了这一部分的概览。

#### A 部分：深度生成建模（DGM）导论 {#part-a-introduction-to-deep-generative-modeling-dgm}

我们从[第 1 章](../chapters/01-deep-generative-modeling/index.md#ch-overview-dgm)开始，回顾深度生成建模的基本目标。给定一组数据样本，目标是建立一个模型，使其能够生成新的样本，而这些新样本看起来来自同一个通常未知的底层数据分布。许多方法通过学习数据的分布方式来实现这一目标：既可以通过概率模型显式地学习，也可以通过学得的变换隐式地学习。随后，我们解释这类模型如何利用神经网络表示数据分布，如何从样本中学习，以及如何生成新样本。本章最后对主要生成框架进行分类，突出它们的核心思想与关键区别。

#### B 部分：扩散模型的核心视角 {#part-b-core-perspectives-on-diffusion-models}

<span id="fig-ill-forward-reverse"></span>

[![从数据到噪声的前向扰动过程，以及从新噪声到新数据样本的逆向生成过程](../../en/assets/front-matter/forward-reverse-diffusion.png)](../../en/assets/front-matter/forward-reverse-diffusion.png)

**图 1：** **前向与逆向扩散轨迹。** 在前向过程（上方）中，一个真实数据样本逐渐受到噪声扰动，经过一系列带噪的中间阶段，最终与纯噪声无法区分。在逆向过程（下方）中，生成从一个新的噪声样本开始，通过逐渐去除噪声产生一幅新图像。逆向过程并非重建训练时见过的某个特定样本，而是从新的噪声开始，逐渐将其塑造成一个逼真的新样本。

来源：作者制作。

在概述深度生成建模的一般目标与机制之后，我们转向扩散模型。这类方法将生成实现为从噪声到数据的逐渐变换（见[图 1](preface.md#fig-ill-forward-reverse)）。其核心思想很简单：通过不断添加噪声，可以逐步扰动一个数据样本，直到它与纯随机噪声无法区分。这一*前向过程*将不同数据样本移入同一个简单的噪声空间；该空间易于采样，并作为生成的起点。在此过程中，它还构造出一系列平滑衔接的带噪中间阶段，将数据与噪声连接起来。生成时，我们从这一空间抽取一个新的噪声样本，通过*逆时间（生成）过程*逐渐去除噪声，最终产生一个新的数据样本。这一逆向过程并不恢复某个特定的原始样本，而是从新的噪声开始，逐渐将其转化为逼真的数据。我们考察三个相互关联的框架，它们都建立在同一原理之上：一个逐渐添加噪声的前向过程，以及一个由一系列逐渐去噪的模型近似的逆时间过程：

-   **变分视角**（[第 2 章](../chapters/02-variational-perspective/index.md#ch-variational)）：源于变分自编码器（VAE） ([Kingma and Welling, 2013](../references.md#ref-kingma2013auto))，将扩散表述为通过变分目标学习去噪过程，由此发展出 Denoising Diffusion Probabilistic Models（DDPM） ([Sohl-Dickstein et al., 2015](../references.md#ref-sohl2015deep); [Ho et al., 2020](../references.md#ref-ho2020denoising))。

-   **基于 score 的视角**（[第 3 章](../chapters/03-score-based-perspective/index.md#ch-score-based)）：源于基于能量的模型（EBM）<br>
     ([Ackley et al., 1985](../references.md#ref-ackley1985learning))，并发展为 Noise Conditional Score Networks（NCSN） ([Song and Ermon, 2019](../references.md#ref-song2019generative))。它学习 score 函数，即数据对数密度的梯度，用于指引如何逐渐去除样本中的噪声。在连续时间下，[第 4 章](../chapters/04-score-sde-framework/index.md#ch-score-sde)介绍了 *Score SDE 框架*，以随机微分方程（SDE）描述这一去噪过程，并以常微分方程（ODE）描述与之对应的确定性过程。这一视角将扩散建模与经典微分方程理论联系起来，为分析和算法设计提供清晰的数学基础。

-   **基于流的视角**（[第 5 章](../chapters/05-flow-based-perspective/index.md#ch-flow-based)）：以归一化流 ([Rezende and Mohamed, 2015](../references.md#ref-rezende2015variational)) 为基础，并由流匹配 ([Lipman et al., 2022](../references.md#ref-lipman2022flow)) 推广。这一视角将生成建模为连续变换，把样本从简单先验传输到数据分布。这一演化由速度场通过 ODE 支配，明确规定概率质量如何随时间移动。这种基于流的表述自然地从先验到数据的生成扩展到更一般的*分布到分布的转换*问题，其目标是学习连接任意一对源分布和目标分布的流。

<span id="fig-timeline-diffusion-perspectives-equal"></span>

[![变分、基于 score 和基于流三种视角的发展时间线，从 1985 年的 EBM 到 2022 年的流匹配](../../en/assets/front-matter/diffusion-perspectives-timeline.png)](../../en/assets/front-matter/diffusion-perspectives-timeline.png)

**图 2：** **现代机器学习中扩散模型各视角的发展时间线。** 每一组使用相同颜色。

- [第 2 章](../chapters/02-variational-perspective/index.md#ch-variational)：变分自编码器（VAE） ([Kingma and Welling, 2013](../references.md#ref-kingma2013auto)) $\to$ Diffusion Probabilistic Models（DPM） ([Sohl-Dickstein et al., 2015](../references.md#ref-sohl2015deep)) $\to$ DDPM ([Ho et al., 2020](../references.md#ref-ho2020denoising))。

- [第 3 章](../chapters/03-score-based-perspective/index.md#ch-score-based)与[第 4 章](../chapters/04-score-sde-framework/index.md#ch-score-sde)：基于能量的模型（EBM） ([Ackley et al., 1985](../references.md#ref-ackley1985learning)) $\to$ Noise Conditional Score Network（NCSN） ([Song and Ermon, 2019](../references.md#ref-song2019generative)) $\to$ Score SDE ([Song et al., 2020](../references.md#ref-song2020score))。

- [第 5 章](../chapters/05-flow-based-perspective/index.md#ch-flow-based)：归一化流（NF） ([Rezende and Mohamed, 2015](../references.md#ref-rezende2015variational)) $\to$ 神经 ODE（NODE） ([Chen et al., 2018](../references.md#ref-chen2018neural)) $\to$ 流匹配（FM） ([Lipman et al., 2022](../references.md#ref-lipman2022flow))。

来源：作者制作。

虽然这些视角乍看之下有所不同，[第 6 章](../chapters/06-unified-perspective/index.md#ch-unified-perspective)表明它们之间存在深刻联系。每种视角都采用一种*条件化策略*，将学习目标转化为可计算的回归问题。在更深层面上，它们都描述了概率分布从先验到数据的同一种时间演化。这一演化由 *Fokker–Planck 方程*支配；该方程可以看作密度在连续时间下的变量替换，确保随机表述与确定性表述之间的一致性。

由于扩散模型可以被视为将一个分布传输到另一个分布的方法，[第 7 章](../chapters/07-optimal-transport/index.md#ch-ot-eot)阐述了它们与经典最优传输以及 Schrödinger 桥之间的联系，后者可解释为带熵正则化的最优传输。我们回顾静态与动态两种表述，并解释它们与连续性方程及 Fokker–Planck 视角的关系。对于侧重实践的读者，本章可作为选读；对于希望进一步研究这些联系的读者，本章则提供严格的数学背景和经典文献指引。

<span id="fig-part-b-roadmap"></span>

[![经典生成模型与三种扩散视角、连续时间表述及统一原理之间的联系](../../en/assets/front-matter/part-b-roadmap.png)](../../en/assets/front-matter/part-b-roadmap.png)

**图 3：** **B 部分：扩散模型的统一与原理视角。** 该图直观地将经典生成建模方法——变分自编码器、基于能量的模型和归一化流——与各自对应的扩散模型表述联系起来。每条纵向路径展示一条概念发展的脉络，最终汇入连续时间框架。三种视角（变分、基于 score 和基于流）提供了各不相同但在数学上等价的解释。

来源：作者制作。

### C、D 两部分：控制与加速扩散采样 {#part-c-d-controlling-and-accelerating-the-diffusion-sampling}

在统一基础原理之后，我们转向利用扩散模型进行高效生成的实践问题。从扩散模型中采样对应于求解一个微分方程。然而，这一过程的计算开销通常很大。C、D 两部分重点讨论如何通过改进采样方法和学习加速技术，提升生成质量、可控性与效率。

#### C 部分：从扩散模型中采样 {#part-c-sampling-from-diffusion-models}

扩散模型的生成过程呈现出鲜明的由粗到细的细化特点：噪声被逐步去除，得到的样本在结构和细节上越来越连贯。这一性质也带来取舍。一方面，它允许精细的控制；通过在学得的、随时间变化的速度场中添加引导项，我们可以调整 ODE 流，使其体现用户意图，从而实现可控采样。另一方面，所需的迭代积分使采样速度慢于一次生成的生成器。本部分重点讨论如何在推断时改进生成过程，而无需重新训练。

-   **引导生成**（[第 8 章](../chapters/08-guidance/index.md#ch-guidance)）**：** 分类器引导和无分类器引导等技术，使生成过程能够以用户定义的目标或属性为条件。在此基础上，我们进一步讨论如何使用偏好数据集，使扩散模型更好地对齐这些偏好。

-   **利用数值求解器实现快速生成**（[第 9 章](../chapters/09-numerical-solvers/index.md#ch-numerical-solvers)）**：** 使用高级数值求解器，以更少的步数近似逆向过程，可以显著加快采样速度，在保持质量的同时降低开销。

#### D 部分：学习快速生成模型 {#part-d-learning-fast-generative-models}

除了改进现有采样算法，我们还研究如何直接学习近似扩散过程的快速生成器。

-   **基于蒸馏的方法**（[第 10 章](../chapters/10-distillation/index.md#ch-distillation)）：这一方法着重训练学生模型，使其模仿预训练的、采样缓慢的扩散模型（教师）的行为。目标并非减小教师模型的规模，而是用少得多的积分步数重现其采样轨迹或输出分布，通常仅需几步，甚至一步。

-   **从头学习**（[第 11 章](../chapters/11-fast-generators/index.md#ch-fast-scratch)）**：** 由于扩散模型的采样可以视为求解 ODE，这一方法不依赖教师模型，直接从头学习解映射（即流映射）。学得的映射可以将噪声直接变为数据，更一般地，也可以沿解轨迹在任意时刻之间跳转。

### 结语与展望：超越连续状态空间上的扩散 {#epilogue-beyond-diffusion-on-continuous-state-spaces}

我们在[第 12 章](../chapters/12-epilogue/index.md#ch-epilogue)结束本书，明确阐述贯穿全书的原理。从根本上说，扩散建模并不局限于某一种噪声类型、某一类模型，甚至不局限于连续状态空间。它研究的是概率质量如何在给定的、随时间变化的前向扰动过程下演化，以及如何将这一演化逆转、近似或用于生成。在连续情形下，这一原理通过变量替换公式、连续性方程和 Fokker–Planck 方程体现出来。在离散情形下，转移核、连续时间 Markov 链和主方程发挥着同样的结构性作用。

最后一章还表明，贯穿本书的三种视角，即变分、基于 score 和基于流的视角，可以扩展到连续数据建模之外。即使对于文本、蛋白质序列及其他基于 token 的对象这类离散数据，也可以通过变分目标、类似 score 的比值或逆向速率刻画，以及类似流的传输视角来表述逆向建模。数学对象发生了变化，但其背后的原理没有改变。因此，结语与展望既在概念上归结全书，也展望了更广阔的生成建模领域。

### 附录 {#appendices}

为了便于所有读者阅读，附录提供了基础概念的背景知识。[附录 A](../appendices/a-differential-equations/index.md#app-de)简要介绍微分方程，它已成为描述扩散模型的语言。

尽管扩散模型的视角和起源各不相同，其背后的核心认识在于*变量替换公式*。这一基础自然延伸到 *Fokker–Planck 方程*和*连续性方程*等更深入的概念，它们描述了在由函数（离散时间）或微分方程（连续时间）定义的映射下，概率密度如何变换与演化。[附录 B](../appendices/b-density-evolution/index.md#app-continuity)以循序渐进的方式介绍这些内容，将基础思想与更高级的概念联系起来。在[附录 C](../appendices/c-ito-calculus/index.md#app-ito)中，我们介绍扩散模型背后两种强大却常被忽视的工具：*Itô 公式*与 *Girsanov 定理*，它们为 Fokker–Planck 方程和逆时间采样过程提供严格支撑。最后，[附录 D](../appendices/d-supplementary-proofs/index.md#app-supplementary-proofs)汇集了正文各章中部分命题与定理的证明。

### 本书涵盖与不涵盖的内容 {#what-this-book-covers-and-what-it-does-not}

我们希望本书的内容具有持久价值。从自上而下的视角看，本书从一条原理出发：构造连续时间动力学，将简单先验传输到数据分布，同时确保每个时刻的边缘分布，都与给定的从数据到噪声的前向过程所诱导的边缘分布一致。由这一原理出发，我们建立实现采样的随机流和确定性流，说明如何引导轨迹（引导），并解释如何加速这一过程（数值求解器）。随后，我们研究受扩散启发的快速生成器，包括蒸馏方法和流映射模型。借助这些工具，读者可以将新论文置于共同框架下理解，明白方法为何有效，并设计改进的模型。

本书不试图对扩散模型文献进行穷尽式综述，也不罗列架构、训练实践或超参数，不比较不同方法的实验结果，不介绍数据集与排行榜，不描述特定领域或模态的应用，不涉及系统层面的部署，不提供大规模训练的操作方案，也不讨论硬件工程。这些主题发展迅速，更适合通过专题综述、开放代码仓库和实现指南加以介绍。

[^1]: 我们对扩散模型三种视角的组织依据是概念关系，而非时间顺序。例如，*基于能量的方法*有着更为悠久的历史。统计力学将概率与能量联系起来，而对 Brownian 运动的研究及其后续工作描述了概率分布如何随时间演化  ([Gibbs, 1902](../references.md#ref-gibbs1902elementary); [Langevin, 1908](../references.md#ref-langevin1908brownian); [Fokker, 1914](../references.md#ref-fokker1914mittlere); [Planck, 1917](../references.md#ref-planck1917satz); [Kolmogoroff, 1931](../references.md#ref-kolmogorov1931))。后来，现代扩散概率模型将相关思想引入生成建模，并明确受到非平衡统计物理的启发  ([Sohl-Dickstein et al., 2015](../references.md#ref-sohl2015deep))。因此，这里介绍的三种视角应被理解为解释扩散模型的现代方式，而不应被视为严格按照其背后思想首次出现的时间排列的历史顺序。
