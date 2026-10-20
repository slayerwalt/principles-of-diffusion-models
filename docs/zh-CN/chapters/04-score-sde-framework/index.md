# 4. 当今的扩散模型：Score SDE 框架 {#ch-score-sde}

> *只有一种精确表述这些定律的方法，那就是使用微分方程。它们的优点在于具有根本性，而且据我们所知是精确的。*
>
> — Richard P. Feynman

到目前为止，我们从两个视角研究了扩散模型：变分视角与基于 score 的视角，后者自然地源自 EBM 的形式。现在，我们进一步转向*连续时间框架*。其核心是 *Score SDE*（[Song et al., 2020](../../references.md#ref-song2020score)），它提供了将 DDPM 与 NCSN 纳入同一框架的连续时间视角。

我们不再只处理有限的噪声水平序列，而是设想在数据与噪声之间插入越来越多的中间水平。在连续时间极限下，噪声水平随时间平滑变化，形成如下基本图景：

$$
p_{\mathrm{data}}
    \;\xrightarrow{\text{forward SDE}}\;
    p_T \approx p_{\mathrm{prior}},
    \qquad
    p_{\mathrm{prior}}
    \;\xrightarrow{\text{reverse dynamics}}\;
    p_{\mathrm{data}}.
$$

前向 SDE 描述数据如何逐渐受到扰动而变成噪声，在连续时间中扮演 DDPM 噪声调度或 NCSN 噪声水平序列的角色。一旦指定了前向过程，主要挑战就是沿相反方向生成：从简单噪声分布出发，如何恢复数据分布？

关键量是随时间变化的 score

$$
\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}}),
$$

它描述每个时刻 $t$ 的含噪分布的局部几何。一旦知道这一 score，或用神经网络近似它，就能确定用于生成的逆时间动力学，将噪声转化为数据。由此得到两种互补描述：随机的*逆时间 SDE* 与确定性的*概率流 ODE*（PF-ODE）。虽然它们的样本轨迹不同，但可以遵循相同的边缘分布演化。

连续时间视角将生成转化为求解微分方程的问题，使我们能够使用随机过程、微分方程与数值分析中的广泛工具。特别是，一旦将生成动力学写成 SDE 或 ODE，就可以使用标准数值方法进行模拟，而更先进的求解器可以提高采样效率和准确性。因此，这一视角既为扩散模型提供统一的数学描述，也为理解和设计本书后续介绍的快速采样方法奠定了实践基础。

**读完本章后，你将能够……**

-   理解 DDPM 与 NCSN 的离散噪声水平如何导向连续时间前向 SDE，使数据逐渐转化为噪声；

-   理解从噪声回到数据的生成过程，既可由随机逆时间 SDE 描述，也可由确定性的概率流 ODE（PF-ODE）描述；

-   解释逆时间 SDE 与 PF-ODE 如何在单个样本轨迹不同的情况下遵循相同的边缘分布序列，并理解 Fokker–Planck 方程如何描述概率密度的连续时间演化；

-   理解如何通过去噪 score 匹配学习随时间变化的 score，再通过逆时间 SDE 或 PF-ODE 将其用于采样；

-   理解 PF-ODE 如何也能沿前向运行以进行反演，并通过连续变量替换公式跟踪概率密度变化，从而计算似然；

-   认识到 VE SDE 与 VP SDE 是同一连续时间扩散框架的两个重要实例，分别联系到 NCSN 与 DDPM。

**阅读路线。** 不熟悉随机微分方程的读者，可按需查阅[附录 A](../../appendices/a-differential-equations/index.md#app-de)中的微分方程内容。首次阅读时，应重点理解前向 SDE、逆时间 SDE 和 PF-ODE 的作用，以及相应的训练与采样过程。在掌握整体思路后，再回顾 Gaussian 示例与选读推导。

继续阅读以下各节及[参考文献](../../references.md#references)。

[4.1 Score SDE 的原理](01-score-sde-principles.md#sec-dm-today)

[4.2 前向／逆时间 SDE 与 PF-ODE 的边缘分布匹配](02-matching-marginals.md#subsec-fpe-sde-ode)

[4.3 Score SDE 的训练与采样](03-training-and-sampling.md#sec-scoresde-training-sampling)

[4.4 SDE 的具体实例](04-sde-instantiations.md#sec-instances)

[4.5 通过累积扰动核的直接视角](05-accumulated-perturbation-kernels.md#sec-variational-perspective)

[4.6 （选读）通过边缘化与 Bayes 法则推导 Fokker–Planck 方程及逆时间 SDE](06-fokker-planck-and-reverse-sde.md#eq-heuristic-fpe-sde)

[4.7 结语](07-closing-remarks.md#sec-ch4-cr)

<span id="dsm-minimizer"></span>

[命题 4.3.1](03-training-and-sampling.md#dsm-minimizer)

<span id="sec-variational-perspective"></span>

[第 4.5 节](05-accumulated-perturbation-kernels.md#sec-variational-perspective)

<span id="reverse-kernel"></span>

[引理 4.5.2](05-accumulated-perturbation-kernels.md#reverse-kernel)