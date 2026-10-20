# 7.1 分布到分布转换引言 {#sec-distribution-translation}

扩散模型将终端分布固定为标准 Gaussian 分布 $p_{\mathrm{prior}}$。然而，许多应用需要*分布到分布*的转换：将源分布 $p_{\mathrm{src}}$ 变为不同的目标分布 $p_{\mathrm{tgt}}$。例如，将草图转换成逼真的照片图像，或在不同艺术风格之间转换。

现代扩散方法为此提供了实用的途径。SDEdit 等*单端点*方法（[Meng et al., 2021](../../references.md#ref-meng2021sdedit)）从 $t=0$ 时刻的源图像出发，将其扩散到中间步 $t$，然后使用针对目标领域预训练的扩散模型逆转这一过程。这会产生风格和内容符合目标分布的输出。

Dual Diffusion Bridge 等*双端点*方法（[Su et al., 2022](../../references.md#ref-su2022dual)）则通过共享潜分布连接两个领域，通常使用 $t=1$ 时刻的 Gaussian 分布。前向概率流 ODE 将 $p_{\mathrm{src}}$ 中的样本传输到这一潜空间，而在目标领域训练的逆向 ODE 则将它们映射回 $p_{\mathrm{tgt}}$。除了这些在采样时进行转换的方法，[第 5.2 节](../05-flow-based-perspective/02-flow-matching-framework.md#sec-flow-matching-framework)介绍的流匹配框架还提供了一种基于训练的替代方案：直接学习一个 ODE 流，将质量从 $p_{\mathrm{src}}$ 连续地移至 $p_{\mathrm{tgt}}$。

关键在于，分布之间的转换需要的不只是两个分别训练的模型。它需要一种有理论依据的映射，使*两个*端点处的动力学相互匹配，并以“最便宜”（代价高效）的方式完成转换。

本节将关注这一经典分布到分布问题的数学基础，不逐一介绍众多基于扩散的转换应用。具体而言，我们着重讨论最优传输（optimal transport，OT）及其熵正则化变体 Schrödinger 桥（Schrödinger bridge，SB）。理论领域长期以来一直将它们视为实现代价高效（数学意义上）分布转换的标准表述。

其根本问题是：

<span id="question-7-1-1"></span>

**问题 7.1.1**

给定两个概率分布，如何在最小化总*代价*的同时，以最高效的方式将一个分布转换为另一个？

这里的代价 $c(\mathbf{x}, \mathbf{y})$ 是一个非负函数，用来度量将单位质量从点 $\mathbf{x}$ 移到点 $\mathbf{y}$ 的惩罚。一个常见选择是距离的平方 $c(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|^2$。

本节通过简要介绍，说明包括流匹配在内的扩散方法如何与经典及正则化最优传输联系起来。我们要探讨的核心问题是：

<span id="question-7-1-2"></span>

**问题 7.1.2**

扩散模型是否是连接 $p_{\text{data}}$ 和 $p_{\text{prior}}$ 的一种最优传输？如果是，具体是在何种意义上？

为回答这一问题，我们首先在[第 7.2 节](02-transport-problem-setups.md#sec-ot-intro)澄清“最优性”的含义。我们回顾经典*最优传输（OT）*的静态 Monge–Kantorovich 形式（[式 (7.2.1)](02-transport-problem-setups.md#eq-ot)）及其动态 Benamou–Brenier 表述（[式 (7.2.3)](02-transport-problem-setups.md#eq-bb-ot)，在连续性方程的约束下最小化动能），以及[式 (7.2.5)](02-transport-problem-setups.md#eq-eot-epsilon)中的熵正则化变体（*熵正则化 OT*），后者等价于 *Schrödinger 桥问题*（[式 (7.2.8)](02-transport-problem-setups.md#eq-sb-kinetic)）。从动态视角看，OT 诱导一个满足连续性方程的确定性流，而 SB 诱导一个受控扩散，其边缘分布按 Fokker–Planck 方程演化。[第 7.3 节](03-relationships-between-formulations.md#sec-relation-ot)将概述这些表述之间的联系。

随后，我们分两部分讨论。首先，[第 7.4 节](04-diffusion-sdes-and-schrodinger-bridges.md#sec-dm-and-sb)说明，标准扩散模型使用的固定前向加噪 SDE 本身并不是任意 $p_{\mathrm{src}}$ 和 $p_{\mathrm{tgt}}$ 之间的 Schrödinger 桥：前向过程是选定的参考扩散，而正时间或逆时间 SDE 一般都不会强制精确匹配给定目标端点。因此，除非显式求解带有这些端点的 SB 问题，否则它不具备熵正则化 OT 意义下的最优性；不过，由于它锚定了一个起始端点，它确实是半桥问题的最优解。

其次，[第 7.5 节](05-probability-flow-odes-and-optimal-transport.md#sec-pfode-ot)回到 $p_{\mathrm{src}}=p_{\mathrm{prior}}$（Gaussian 分布）和 $p_{\mathrm{tgt}}=p_{\mathrm{data}}$ 的生成设定。按照构造，PF-ODE 定义了一个将 $p_{\mathrm{prior}}$ 传输到 $p_{\mathrm{data}}$ 的确定性映射。然而，对于给定传输代价（如二次代价 $W_2$），该流通常不是 OT 映射：它实现了众多可行确定性耦合中的一个，并不最小化 Benamou–Brenier 作用量。接下来，我们讨论“流矫正”过程（[第 5.4.1 节](../05-flow-based-perspective/04-canonical-affine-flow.md#subsec-rectify)）能否得到 OT 映射；但一般而言，并没有这样的理论保证。因此，精确刻画扩散模型的 PF-ODE 映射与 OT 之间的关系，仍然是一个具有挑战性且尚未解决的问题。
