# 9. 用于快速扩散采样的数值求解器 {#ch-numerical-solvers}

扩散模型将噪声映射为数据样本的生成过程，可以通过逆时间 SDE 或相应 PF-ODE 表述。这一过程本身较慢，因为它依赖数值求解器，通过许多小积分步近似解轨迹（简要介绍见[附录 A](../../appendices/a-differential-equations/index.md#app-de)）。因此，加速推断已成为核心研究目标。总体而言，现有方法分为两类：

-   **免训练方法：** 本章的重点。这些方法开发先进数值求解器，无需额外训练即可提高扩散采样效率。

-   **基于训练的方法：** 将在[第 10 章](../10-distillation/index.md#ch-distillation)和[第 11 章](../11-fast-generators/index.md#ch-fast-scratch)介绍。这些技术将预训练扩散模型蒸馏为快速生成器，或直接学习 ODE 流映射（解），使采样只需少量步骤。

基于 SDE 的采样器（如 Euler–Maruyama）由于具有随机性，可能产生更多样的样本，但通常需要更多步数（[Xu et al., 2023](../../references.md#ref-xu2023restart)）。这里关注基于 ODE 的生成，其原理可以自然扩展到 SDE 情形。

**读完本章后，你将能够……**

-   将扩散采样视为 PF-ODE 的数值积分问题；

-   区分模型近似误差与数值离散化误差；

-   理解 DDIM、DEIS、DPM-Solver 和 DPM-Solver`++` 背后的主要数值思想；

-   将现代扩散采样器与经典数值方法联系起来，包括 Euler、Runge–Kutta（RK）、Adams–Bashforth（AB）及指数积分器。

**阅读路线。** 不熟悉微分方程或数值求解器的读者，可先参阅[附录 A](../../appendices/a-differential-equations/index.md#app-de)。阅读本章时，始终记住一个问题：每个求解器精确积分了什么，又对什么进行了数值近似？

继续阅读下面各节及[参考文献](../../references.md#references)。

[9.1 引言](01-prologue.md#sec-prologue-solvers)

<span id="sec-revisit-ddim"></span>

[9.2 DDIM](02-ddim.md#sec-revisit-ddim)

[9.3 DEIS](03-deis.md#sec-exponential)

[9.4 DPM-Solver](04-dpm-solver.md#sec-dpm)

[9.5 DPM-Solver`++`](05-dpm-solver-plus-plus.md#sec-dpm)

[9.6 （选读）ParaDiGMs](06-paradigms.md#sec-picard)

[9.7 结语](07-closing-remarks.md#sec-ch9-cr)

<span id="eq-ddim-update-all"></span>

[式 (9.2.2)：确定性 DDIM 更新的等价形式](02-ddim.md#eq-ddim-update-all)。

<span id="subsec-ddim"></span>

[9.2.3 （选读）DDIM 最初的变分视角](02-ddim.md#subsec-ddim)。
