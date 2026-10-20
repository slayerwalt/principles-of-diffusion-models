# 7.4 扩散模型的 SDE 是 SB 问题的最优解吗？ {#sec-dm-and-sb}

## 7.4.1 扩散模型是 Schrödinger 桥的一个特例 {#subsec-diffusion-special-case-sb}

SB 框架通过实现任意源分布与目标分布之间的非线性插值，扩展了（基于 score 的）扩散模型。它加入由标量势函数 $\Psi(\mathbf{x}, t)$ 和 $\widehat{\Psi}(\mathbf{x}, t)$ 导出的控制漂移项，引导参考扩散过程匹配给定端点的边缘分布（见[式 (7.2.12)](02-transport-problem-setups.md#eq-pde-schrodinger)），并遵循如下分解：

$$
\nabla\log{\Psi(x,t)}+\nabla\log{\hat{\Psi}(x,t)}=\nabla \log p_t^{\mathrm{SB}}(\mathbf{x}).
$$

这一推广使模型可以超出标准 Gaussian 先验的限制，从更广泛的分布中生成样本。

**与扩散模型的联系。**

扩散模型是 SB 框架的一个特例。假设势函数为常数，即 $\Psi(\mathbf{x}, t) \equiv 1$。在这一假设下，[式 (7.2.12)](02-transport-problem-setups.md#eq-pde-schrodinger)中的第二个 PDE 化为标准 Fokker–Planck 方程，其解是参考过程的边缘密度：

<span id="eq-sb-backward-potential"></span>

$$
\begin{aligned}
\widehat{\Psi}(\mathbf{x}, t) = p_t^{\mathrm{SB}}(\mathbf{x}).
\end{aligned}
\tag{7.4.1}
$$

相应的 SB 前向 SDE 因而变为未受控的参考过程：

$$
\mathop{}\!\mathrm{d}\mathbf{x}_t = \mathbf{f}(\mathbf{x}_t, t)\mathop{}\!\mathrm{d} t + g(t)\mathop{}\!\mathrm{d}\mathbf{w}_t,
$$

而 SB 后向 SDE 简化为：

$$
\mathop{}\!\mathrm{d}\mathbf{x}_t = \left[\mathbf{f}(\mathbf{x}_t, t) - g^2(t) \nabla \log p_t^{\mathrm{SB}}(\mathbf{x}_t) \right]\mathop{}\!\mathrm{d} t + g(t)\mathop{}\!\mathrm{d}\bar{\mathbf{w}}_t,
$$

这与扩散模型使用的 Anderson 逆时间 SDE 一致。这种对应关系说明，扩散模型可解释为 SB 的零控制极限，此时势函数不引入附加漂移。

**边界条件与一般性。** 除非边界约束相容，否则上述化简只是形式上的。对于任意源／目标分布 $(p_{\mathrm{src}},p_{\mathrm{tgt}})$，选择 $\Psi\equiv 1$ 通常无法满足[式 (7.2.12)](02-transport-problem-setups.md#eq-pde-schrodinger)中的 PDE 边界条件。完整 SB 通过学习非平凡势函数解决这一问题；这些势函数诱导非线性控制漂移，使参考动力学发生弯曲，从而匹配任意给定端点。扩散模型则将一个端点固定为简单先验（通常为 Gaussian 分布），并仅学习逆时间 score 以到达数据。由此看，SB 是一个更灵活的统一框架：使用非平凡势函数时，它连接任意端点；当 $\Psi\equiv 1$ 时，它退化为上述扩散模型情形。另外，在标准线性扩散模型中，$p_T \approx p_{\mathrm{prior}}$ 仅在 $T \to \infty$ 时成立，因此与先验的匹配只是近似的。

## 7.4.2 扩散模型作为 Schrödinger 半桥 {#subsec-schrodinger-half-bridges}

本节解释为什么扩散模型不是完整的 Schrödinger 桥，而可以通过放宽后的 *Schrödinger 半桥*概念理解。半桥只施加一个端点约束（$p_{\mathrm{prior}}$ 或 $p_{\mathrm{data}}$），因此是完整桥的单侧变体。在形式化这一联系之前，我们先基于[式 (7.2.7)](02-transport-problem-setups.md#eq-sb-epsilon-general)中采用任意 $p_{\mathrm{src}}$ 和 $p_{\mathrm{tgt}}$ 的一般表述，定义 Schrödinger 半桥。随后回到扩散模型，说明当端点为 $p_{\mathrm{prior}}$ 和 $p_{\mathrm{data}}$ 时，半桥视角如何自然地适用。

**Schrödinger 半桥**

SB 问题寻找一个随机过程，其分布在 KL散度意义下最接近简单参考过程，同时*匹配两个端点分布* $p_{\mathrm{src}}$ 和 $p_{\mathrm{tgt}}$。求解完整桥需要同时满足两个边界条件，计算上往往很困难。一种有用的放宽是*半桥*问题：只匹配其中一个端点。

形式上，令 $R$ 为参考路径分布。*前向半桥*寻找使下式最小的路径分布 $P$：

$$
\min_{P: P_0 = p_{\mathrm{src}}}
    \mathcal{D}_{\mathrm{KL}}(P \,\|\, R),
$$

只要求满足约束 $P_0 = p_{\mathrm{src}}$。类似地，*后向半桥*只约束终端分布，

$$
\min_{P: P_T = p_{\mathrm{tgt}}}
    \mathcal{D}_{\mathrm{KL}}(P \,\|\, R).
$$

换言之，前向半桥询问：在所有从所需初始分布出发的过程中，哪一个最像参考过程？后向半桥则针对终止于所需终端分布的过程询问同一问题。通过迭代组合这两种放宽，可以近似完整 SB。

**扩散模型未实现精确端点匹配。** 扩散模型与 SB 框架的一项关键区别，在于如何处理终端分布 $p_T$。标准扩散模型的前向 SDE 通常关于 ${\mathbf{x}}_t$ 线性（见[式 (4.4.2)](../04-score-sde-framework/04-sde-instantiations.md#eq-forward-linear-sde)），并设计为使 $p_T$ 仅在 $T \to \infty$ 时逼近先验：

$$
p_T \approx p_{\mathrm{prior}}.
$$

但在有限时间，$p_T$ 是 $\alpha_T{\mathbf{x}}_0+\sigma_T{\boldsymbol{\epsilon}}$ 的分布，因此一般仍依赖 $p_{\mathrm{data}}$；它不一定与给定 Gaussian 先验精确一致（见[第 C.1.5 节](../../appendices/c-ito-calculus/01-ito-formula.md#subsec-rigorous-proof-linear-sde)）。

SB 框架则在有限时刻 $T$ 强制实现精确边缘匹配，做法是引入形如 $g^2(t)\nabla_\mathbf{x}\log \Psi(\mathbf{x}, t)$ 的附加控制漂移。这保证终端分布精确满足 $p_T = p_{\mathrm{prior}}$，无论初始数据分布 $p_0 = p_{\mathrm{data}}$ 是什么。概括如下：

-   **扩散模型：** $p_T \approx p_{\mathrm{prior}}$，在 $T \to \infty$ 时渐近成立；

-   **Schrödinger 桥：** $p_T = p_{\mathrm{prior}}$，在有限 $T$ 处精确成立，这是通过求解控制势函数 $\Psi$ 和 $\widehat{\Psi}$ 实现的。

**Diffusion Schrödinger Bridge。** 固定的前向加噪过程可视为相对于选定参考动力学、锚定于 $p_{\mathrm{data}}$ 的前向 Schrödinger 半桥。其终端边缘分布由该参考过程产生，一般只是近似于 $p_{\mathrm{prior}}$；因此，它不是给定端点对 $(p_{\mathrm{data}},p_{\mathrm{prior}})$ 之间的完整桥。

为处理这一问题，Diffusion Schrödinger Bridge（DSB）（[De Bortoli et al., 2021](../../references.md#ref-de2021diffusion)）沿用迭代比例拟合（Iterative Proportional Fitting，IPF）算法的思想，交替匹配两个端点的边缘分布。IPF 是一种交替投影方法。它按如下方式扩展扩散模型，以求解完整 SB 问题[^1]：

-   **步骤 0：参考过程。** 用 $P^{(0)} := R_{\text{fwd}}$ 初始化，即参考前向 SDE：

    $$
    \mathrm{d} \mathbf{x}_t = \mathbf{f}(\mathbf{x}_t, t)   \mathrm{d} t + g(t)   \mathrm{d} \mathbf{w}_t, \quad \mathbf{x}_0 \sim p_{\text{data}}.
    $$

    这保证 $P_0^{(0)} = p_{\text{data}}$，但通常 $P_T^{(0)} \neq p_{\text{prior}}$。

-   **步骤 1：后向步骤。** 计算过程 $P^{(1)}$，使其匹配 $p_{\text{prior}}$（时刻 $T$），同时接近 $P^{(0)}$：

    $$
    P^{(1)} = \operatorname*{arg\,min}_{P : P_T = p_{\text{prior}}} \mathcal{D}_{\text{KL}}(P  \|  P^{(0)}).
    $$

    通过神经网络 $\mathbf{s}_{\boldsymbol{\phi}^\times}$ 近似 oracle score 函数即可实现，得到逆时间 SDE：

    $$
    \mathrm{d} \mathbf{x}_t = \left[ \mathbf{f}(\mathbf{x}_t, t) - g^2(t) \mathbf{s}_{\boldsymbol{\phi}^\times}(\mathbf{x}_t, t) \right] \mathrm{d} t + g(t)  \mathrm{d} \bar{\mathbf{w}}_t,
    $$

    从 $\mathbf{x}_T \sim p_{\text{prior}}$ 开始向后模拟。

-   **迭代。** 过程 $P^{(1)}$ 满足 $P_T^{(1)} = p_{\text{prior}}$，但其初始边缘分布 $P_0^{(1)}$ 通常偏离 $p_{\text{data}}$。IPF 通过学习一个前向 SDE，将 $P_0^{(1)}$ 调整回 $p_{\text{data}}$，然后再进行后向步骤以强制满足 $p_{\text{prior}}$。如此交替，不断改进过程，直至收敛到最优桥 $P^*$，它同时满足 $P_0^* = p_{\text{data}}$ 和 $P_T^* = p_{\text{prior}}$。[De Bortoli et al. (2021)](../../references.md#ref-de2021diffusion) 在较弱条件下证明了收敛性。

上述 DSB 表述在发展历程中具有重要地位，概念也很清晰，但实际实现的计算开销可能很大。前向与后向投影通过迭代进行，因此每次 IPF 迭代都需要求解一个独立的 score 匹配问题，而且近似误差可能跨轮累积。更广泛地，越来越多的研究尝试通过发展稳定性和可扩展性更好的 Schrödinger 桥表述来缓解这些困难，有些方法还减少了在整个迭代过程中反复拟合独立扩散模型的需要。这里不详细讨论这些进展，只为感兴趣的读者指出更广泛的实用 SB 方法文献（[Shi et al., 2023](../../references.md#ref-shi2023dsbm); [Liu et al., 2024](../../references.md#ref-liu2024gsbm)）。

[^1]: 虽然这里使用 $p_{\mathrm{data}}$ 和 $p_{\mathrm{prior}}$ 进行说明，但 DSB 框架适用于任意一对端点分布。
