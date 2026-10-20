# B. 密度演化：从变量变换到 Fokker–Planck 方程 {#app-continuity}

理解概率密度如何在变换下演化，是概率论和生成建模的基础。特别地，扩散模型旨在构造生成过程，使其诱导的密度路径逆转预先定义的前向过程。这种演化由连续性方程支配，或在随机情形下由 Fokker–Planck 方程支配。

尽管这些名称可能显得陌生或令人畏惧，它们实际上是基础微积分中变量变换公式的连续时间对应形式。[第 B.1 节](01-change-of-variable.md#sec-cov)从确定性双射出发，逐步介绍一系列变量变换公式，最终到达随机微分方程。这一过程自然连接离散映射与连续时间流动力学。[图 B.1](index.md#fig-unified-framework-cov)概括了这一统一框架。

[第 B.2 节](02-continuity-equation-intuition.md#sec-intuition-continuity)给出连续性方程的物理和直观解释，强调其与动力系统中密度守恒的联系。

**学完本章后，你将能够……**

-   将变量变换公式与 ODE 的连续性方程及 SDE 的 Fokker–Planck 方程联系起来；

-   将连续性方程理解为概率质量守恒，并区分粒子运动与密度演化；

-   区分输运的 Lagrangian 与 Eulerian 视角，并理解为何仅凭密度路径未必能确定唯一速度场；

-   理解为什么 SDE 及其概率流 ODE 可以具有相同的边缘密度，却具有不同的样本路径。

**阅读路线。** 关键原则是：变量变换公式描述样本移动如何改变其概率分布。本附录从确定性映射出发，发展到连续性方程和 Fokker–Planck 方程，并将 Wasserstein 梯度流作为分布层面训练的选读扩展。

<span id="fig-unified-framework-cov"></span>

![从一个双射到多个双射、ODE 流和 SDE 流，以及相应的密度演化方程](../../../en/assets/appendix-b/change-of-variables-overview.png)

**图 B.1：** 统一的*变量变换公式*。从上到下依次为：（1）单个双射；（2）多个双射的复合；（3）由 ODE 支配的连续时间确定性流及相应的连续性方程；（4）由 SDE 建模的随机流及相应的 Fokker–Planck 方程。

来源：作者绘制。

请继续阅读以下各节及[参考文献](../../references.md#references)。

[B.1 变量变换公式：从确定性映射到随机流](01-change-of-variable.md#sec-cov)

[B.2 连续性方程的直观理解](02-continuity-equation-intuition.md#sec-intuition-continuity)

[B.3 （选读）作为分布层面训练的 Wasserstein 梯度流](03-wasserstein-gradient-flows.md#sec-wgf-distribution-level-training)


<span id="subsec-cov-continuity-eq"></span>

[第 B.1.2 节](01-change-of-variable.md#subsec-cov-continuity-eq)

<span id="eq-cov-multi-layer-nolog"></span>

[公式 (B.1.2)](01-change-of-variable.md#eq-cov-multi-layer-nolog)

<span id="sec-wgf-distribution-level-training"></span>

[第 B.3 节](03-wasserstein-gradient-flows.md#sec-wgf-distribution-level-training)
