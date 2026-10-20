# 3.1 基于能量的模型 {#sec-ebm}

对于已经熟悉 EBM 的读者，本节旨在简要回顾相关内容，并衔接到扩散模型基于 score 的视角。

## 3.1.1 使用能量函数建模概率分布 {#modeling-probability-distributions}

令 $\mathbf{x} \in \mathbb{R}^D$ 表示一个数据点。EBM 通过能量函数 $E_{\boldsymbol{\phi}}(\mathbf{x})$ 定义概率密度，该函数由 ${\boldsymbol{\phi}}$ 参数化，为更可能出现的配置赋予更低的能量。得到的分布为

$$
\begin{aligned}
p_{{\boldsymbol{\phi}}}(\mathbf{x}) := \frac{\exp(-E_{\boldsymbol{\phi}}(\mathbf{x}))}{Z_{\boldsymbol{\phi}}}, \quad
    Z_{\boldsymbol{\phi}} := \int_{\mathbb{R}^D} \exp(-E_{\boldsymbol{\phi}}(\mathbf{x})) \mathop{}\!\mathrm{d}\mathbf{x},
\end{aligned}
$$

其中 $Z_{\boldsymbol{\phi}}$ 称为*配分函数*，用于保证归一化：

$$
\int_{\mathbb{R}^D} p_{{\boldsymbol{\phi}}}(\mathbf{x})  \mathop{}\!\mathrm{d}\mathbf{x} = 1.
$$

<span id="fig-ebm-graph"></span>

![基于能量的模型在训练中重新分配数据区域与非数据区域之间的密度](../../../en/assets/chapter-03/ebm-training.png)

**图 3.1：** **EBM 训练示意图。** 模型降低“差”数据点处的密度（提高能量，红色箭头），并提高“好”数据点处的密度（降低能量，绿色箭头）。

来源：作者绘制。

在这一视角下，能量较低的点对应较高的概率，就像小球滚入山谷一样。配分函数 $Z_{{\boldsymbol{\phi}}}$ 保证所有概率之和为一，因此只有能量的*相对*值才起作用。例如，给所有能量加上同一个常数，会使分子与分母乘以同一个因子，从而不改变分布。

此外，由于配分函数 $Z_{{\boldsymbol{\phi}}}$ 强制概率之和为一，数学上可知，降低某个区域内的能量会增大该区域的概率，而其补集的概率相应减小。因此，EBM 遵循严格的全局权衡：让一个山谷变深，必然会让其他山谷相对变浅；概率质量在整个空间内重新分配，而非独立地赋给各个区域。

**EBM 最大似然训练的挑战。** 原则上，EBM 可以通过最大似然进行训练，自然地平衡数据拟合与全局正则化（见[式 (1.1.2)](../01-deep-generative-modeling/01-what-is-deep-generative-modeling.md#eq-mle)）：

<span id="eq-mle-ebm"></span>

$$
\begin{aligned}
\mathcal{L}_{\text{MLE}}({\boldsymbol{\phi}})
&= \mathbb{E}_{p_{\text{data}}({\mathbf{x}})}  \left[ \log \frac{\exp(-E_{{\boldsymbol{\phi}}}({\mathbf{x}}))}{Z_{\boldsymbol{\phi}}} \right] \\
&= -  \underbrace{\mathbb{E}_{p_{\text{data}}}[E_{{\boldsymbol{\phi}}}({\mathbf{x}})]}_{\text{lowers energy of data}}
   -  \underbrace{\log \int \exp(-E_{{\boldsymbol{\phi}}}({\mathbf{x}}))\mathop{}\!\mathrm{d}{\mathbf{x}}}_{\text{global regularization}},
\end{aligned}
\tag{3.1.1}
$$

其中 $Z_{\boldsymbol{\phi}} = \int \exp(-E_{\boldsymbol{\phi}}({\mathbf{x}}))\mathop{}\!\mathrm{d}{\mathbf{x}}$。第一项降低真实数据的能量，第二项则通过配分函数强制归一化。

然而，在高维空间中，计算 $\log Z_{{\boldsymbol{\phi}}}$ 及其梯度在计算上不可行，因为这需要求模型分布下的期望。这促使人们采用其他目标：或者近似这一项，例如对比散度（[Hinton, 2002](../../references.md#ref-hinton2002training)），或者通过 *score 匹配*完全避开它。

接下来，我们先在[第 3.1.2 节](01-energy-based-models.md#subsec-motivation-score)介绍 score 函数的概念，在[第 3.1.3 节](01-energy-based-models.md#subsec-training-ebm)介绍 score 匹配这一绕过配分函数的可计算训练目标，再在[第 3.1.4 节](01-energy-based-models.md#subsec-sampling-ebm)讨论利用 score 函数进行实际采样的方法——Langevin 动力学。

## 3.1.2 动机：什么是 score？ {#subsec-motivation-score}

对于密度 $p({\mathbf{x}})$（定义在 $\mathbb{R}^D$ 上），*score 函数*是对数密度的梯度：

$$
{\mathbf{s}}({\mathbf{x}}) := \nabla_{{\mathbf{x}}} \log p({\mathbf{x}}), \qquad {\mathbf{s}} \colon \mathbb{R}^D \to \mathbb{R}^D.
$$

直观上，score 构成一个指向更高概率区域的向量场，为数据最有可能出现的位置提供局部指引（见[图 3.2](01-energy-based-models.md#fig-score-function)）。

<span id="fig-score-function"></span>

![score 向量场指向概率密度增大的方向](../../../en/assets/chapter-03/score-vector-fields.png)

**图 3.2：** **score 向量场示意图。** score 向量场 $\nabla_{{\mathbf{x}}} \log p({\mathbf{x}})$ 表示密度增大的方向。

来源：作者绘制。

**为什么建模 score 而非密度？** 对 score 建模具有理论与实践两方面的优势：

**1. 无须归一化常数。** 许多分布只能通过未归一化密度 $\tilde{p}({\mathbf{x}})$ 来定义，例如 EBM 中的 $\exp(-E_{\boldsymbol{\phi}}({\mathbf{x}}))$：

$$
p({\mathbf{x}}) = \frac{\tilde{p}({\mathbf{x}})}{Z},
    \qquad
    Z = \int \tilde{p}({\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}.
$$

尽管计算 $Z$ 在计算上不可行，score 却只依赖于 $\tilde{p}$：

<span id="eq-score-compute"></span>

$$
\begin{aligned}
\nabla_{{\mathbf{x}}}\log p({\mathbf{x}}) = \nabla_{{\mathbf{x}}}\log \tilde{p}({\mathbf{x}}) - \underbrace{\nabla_{{\mathbf{x}}}\log Z}_{=0} = \nabla_{{\mathbf{x}}}\log \tilde{p}({\mathbf{x}}),
\end{aligned}
\tag{3.1.2}
$$

因为 $Z$ 相对于 ${\mathbf{x}}$ 是常数。这完全绕过了配分函数。

**2. 完整的表示。** score 函数能够完整刻画底层分布。由于它是对数密度的梯度，可以通过下式恢复密度（相差一个常数）：

$$
\log p({\mathbf{x}})
= \log p({\mathbf{x}}_0) + \int_0^1 {\mathbf{s}}({\mathbf{x}}_0 + t({\mathbf{x}} - {\mathbf{x}}_0))^\top ({\mathbf{x}} - {\mathbf{x}}_0)  \mathop{}\!\mathrm{d} t,
$$

其中 ${\mathbf{x}}_0$ 是参考点，$\log p({\mathbf{x}}_0)$ 由归一化条件确定。因此，对 score 建模与直接对 $p({\mathbf{x}})$ 建模具有同样的表达能力，而在生成建模中往往更易计算。

## 3.1.3 通过 score 匹配训练 EBM {#subsec-training-ebm}

在 EBM 中，密度定义为 $p_{{\boldsymbol{\phi}}}(\mathbf{x}) = \tfrac{\exp(-E_{\boldsymbol{\phi}}(\mathbf{x}))}{Z_{\boldsymbol{\phi}}}$。最大似然训练需要计算 $Z_{\boldsymbol{\phi}}$，这通常在计算上不可行。一个关键观察是，$p_{{\boldsymbol{\phi}}}$ 的模型 score 可简化为 $-\nabla_{{\mathbf{x}}}E_{\boldsymbol{\phi}}({\mathbf{x}})$，与 $Z_{\boldsymbol{\phi}}$ 无关（见[式 (3.1.2)](01-energy-based-models.md#eq-score-compute)）。

*score 匹配*（[Hyvärinen and Dayan, 2005](../../references.md#ref-hyvarinen2005estimation)）利用了 score 只依赖能量函数这一事实。它通过使模型 score 与未知的数据 score 对齐来训练 EBM，无须拟合归一化概率：

<span id="eq-ebm-sm"></span>

$$
\begin{aligned}
\mathcal{L}_{\mathrm{SM}}(\boldsymbol{\phi})
= \tfrac{1}{2}  \mathbb{E}_{p_{\mathrm{data}}(\mathbf{x})}
\big\| \nabla_{\mathbf{x}} \log p_{\boldsymbol{\phi}}(\mathbf{x})
- \nabla_{\mathbf{x}} \log p_{\mathrm{data}}(\mathbf{x}) \big\|_2^2.
\end{aligned}
\tag{3.1.3}
$$

虽然无法直接获得数据 score，但通过分部积分，可以得到一个仅涉及能量及其导数的等价表达式（详见[命题 3.2.1](02-score-based-models.md#sm-trace)）：

$$
\mathcal{L}_{\mathrm{SM}}(\boldsymbol{\phi}) =
\mathbb{E}_{p_{\mathrm{data}}(\mathbf{x})} \left[
-\operatorname{Tr} \big(\nabla_{\mathbf{x}}^2 E_{\boldsymbol{\phi}}({\mathbf{x}})\big)
+ \tfrac{1}{2}  \|\nabla_{\mathbf{x}} E_{\boldsymbol{\phi}}({\mathbf{x}})\|_2^2
\right] + C,
$$

其中 $\nabla_{\mathbf{x}}^2 E_{\boldsymbol{\phi}}({\mathbf{x}})$ 是 $E_{\boldsymbol{\phi}}$ 的 Hessian 矩阵，$C$ 是与 $\boldsymbol{\phi}$ 无关的常数。

这一形式的吸引力在于，它消除了配分函数，并且训练时无须从模型中采样。其主要缺点是需要二阶导数：尽管只需要 Hessian 矩阵的迹（$\operatorname{Tr} \big(\nabla_{\mathbf{x}}^2 E_{\boldsymbol{\phi}}\big)$），而不需要整个 Hessian 矩阵，但精确计算这一项通常仍需汇总各输入坐标上的二阶导数信息，因此其成本通常随输入维数 $D$ 增长。本章后面将再次讨论如何解决这一限制。

## 3.1.4 使用 score 函数进行 Langevin 采样 {#subsec-sampling-ebm}

对于由能量函数 $E_{\boldsymbol{\phi}}(\mathbf{x})$ 定义的 EBM，可以使用 *Langevin 动力学*进行采样。我们先介绍离散时间的 Langevin 更新，再介绍其连续时间极限，即一个随机微分方程（SDE）。最后讨论 Langevin 动力学如何有效探索复杂能量景观背后的物理直觉。

<span id="fig-langevin"></span>

![Langevin 轨迹沿 score 指向高密度区域](../../../en/assets/chapter-03/langevin-sampling.png)

**图 3.3：** **Langevin 采样示意图。** Langevin 采样使用 score 函数 $\nabla_{\mathbf{x}} \log p_{\boldsymbol{\phi}}(\mathbf{x})$，通过[式 (3.1.5)](01-energy-based-models.md#eq-discrete-langevin-score)的更新（以箭头表示），引导轨迹移向高密度区域。

来源：作者绘制。

**离散时间 Langevin 动力学。** 离散时间的 Langevin 更新为

<span id="eq-discrete-langevin"></span>

$$
\begin{aligned}
\mathbf{x}_{n+1} = \mathbf{x}_n - \eta \nabla_{\mathbf{x}} E_{\boldsymbol{\phi}}(\mathbf{x}_n) +  \sqrt{2 \eta} \boldsymbol{\epsilon}_n, \quad n=0,1,2,\ldots,
\end{aligned}
\tag{3.1.4}
$$

其中 $\mathbf{x}_0$ 从某个分布（通常是 Gaussian 分布）初始化，$\eta > 0$ 是步长，$\boldsymbol{\epsilon}_n \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 是 Gaussian 噪声。噪声通过引入随机性，使探索能够超越局部极小值。

由于 score 函数可以计算为

$$
\nabla_{\mathbf{x}} \log p_{\boldsymbol{\phi}}(\mathbf{x}) = -\nabla_{\mathbf{x}} E_{\boldsymbol{\phi}}({\mathbf{x}}).
$$

更新可等价地写为

<span id="eq-discrete-langevin-score"></span>

$$
\begin{aligned}
\mathbf{x}_{n+1} = \mathbf{x}_n + \eta \nabla_{\mathbf{x}} \log p_{\boldsymbol{\phi}}(\mathbf{x}_n) + \sqrt{2 \eta}\boldsymbol{\epsilon}_n,
\end{aligned}
\tag{3.1.5}
$$

其中 score 函数引导样本移向高密度区域。这一形式是扩散模型的核心，后文将详细介绍。

**连续时间 Langevin 动力学。** 当步长 $\eta$ 趋于零时，离散 Langevin 更新自然收敛到由 *Langevin 随机微分方程（SDE）*[^1]描述的连续时间过程：

<span id="eq-continuous-langevin"></span>

$$
\begin{aligned}
\mathop{}\!\mathrm{d} \mathbf{x}(t) = \nabla_{\mathbf{x}} \log p_{\boldsymbol{\phi}}(\mathbf{x}(t)) \mathop{}\!\mathrm{d} t + \sqrt{2} \mathop{}\!\mathrm{d} \mathbf{w}(t),
\end{aligned}
\tag{3.1.6}
$$

其中 $\mathbf{w}(t)$ 表示标准 Brownian 运动（也称为 Wiener 过程[^2]）。需要理解的是，[式 (3.1.4)](01-energy-based-models.md#eq-discrete-langevin)中的离散更新规则就是该连续 SDE 的 Euler–Maruyama 离散化。

在标准正则性假设下（例如，$p_{\boldsymbol{\phi}}\propto e^{-E_{\boldsymbol{\phi}}}$，其中 $E_{\boldsymbol{\phi}}$ 具有束缚性且足够光滑），$\mathbf{x}(t)$ 的分布以指数速度收敛到 $p_{\boldsymbol{\phi}}$（当 $t\to\infty$ 时）；因此，我们可以通过模拟（求解）[式 (3.1.6)](01-energy-based-models.md#eq-continuous-langevin)中的 SDE 进行采样。

**为什么使用 Langevin 采样？** 从物理角度理解 Langevin 采样很自然：能量函数 $E_{\boldsymbol{\phi}}(\mathbf{x})$ 定义了一个势能景观，决定粒子的行为。根据 Newton 动力学，在这一能量导出的力场中，粒子的运动由以下常微分方程（ODE）描述：

$$
\mathop{}\!\mathrm{d} \mathbf{x}(t) = -\nabla_{\mathbf{x}} E_{\boldsymbol{\phi}}\big(\mathbf{x}(t)\big)  \mathop{}\!\mathrm{d} t,
$$

它确定性地驱动粒子沿坡而下，移向能量函数的局部极小值。然而，这种确定性动力学可能陷入局部极小值，无法探索完整的数据分布。

为克服这一限制，Langevin 动力学引入随机扰动，得到 SDE

$$
\mathop{}\!\mathrm{d} \mathbf{x}(t) = -\nabla_{\mathbf{x}} E_{\boldsymbol{\phi}}\big(\mathbf{x}(t)\big)  \mathop{}\!\mathrm{d} t + \underbrace{\sqrt{2 } \mathop{}\!\mathrm{d} {\mathbf{w}}(t)}_{\text{injected noise}},
$$

其中 ${\mathbf{w}}(t)$ 是标准 Brownian 运动。

噪声项使粒子能够跨越能量势垒、逃离局部极小值，从而使轨迹成为一个随机过程，其平稳分布收敛到 Boltzmann 分布

$$
p_{\boldsymbol{\phi}}(\mathbf{x}) \propto e^{-E_{\boldsymbol{\phi}}(\mathbf{x})}.
$$

从这一角度看，EBM 可以理解为学习一个将样本推向高概率区域的力场。Langevin 采样尤其适合 EBM，因为它提供了从模型分布 $p_{\boldsymbol{\phi}}(\mathbf{x})$ 生成样本的实用方法，无须显式计算配分函数。通过反复执行 Langevin 更新，可以获得近似服从目标分布的样本。

**Langevin 采样的固有挑战。** Langevin 动力学是一种广泛使用的基于 MCMC 的采样器，但在高维空间中存在严重限制。其效率高度依赖步长 $\eta$、噪声尺度的选择，以及准确逼近目标分布所需的迭代次数。

这种低效的核心在于“混合时间”过长：对于具有许多孤立模式的复杂数据分布，Langevin 采样往往需要极长时间才能在各个高概率区域之间转移。随着维数增加，这一问题显著恶化，导致收敛慢到难以接受。

可以将采样看作探索一片广阔而崎岖的地形，其中有许多相距遥远的山谷，每个山谷对应一种数据模式。Langevin 动力学依赖局部随机更新，难以高效地在这些山谷之间穿行。因此，它往往无法捕捉分布的全部多样性。

这种低效表明，需要更具结构、更有引导性的采样方法，以便比纯粹随机探索更有效地遍历复杂的数据流形。

[^1]: 使用系数 $\sqrt{2}$ 时，Langevin 动力学使 $p_{\boldsymbol\phi}$ 不随时间改变。也就是说，$p_{\boldsymbol\phi}$ 是平稳的：若 ${\mathbf{x}}(0)\sim p_{\boldsymbol\phi}$，则 ${\mathbf{x}}(t)\sim p_{\boldsymbol\phi}$ 对所有 $t\ge 0$ 都成立。等价地，$p_{\boldsymbol\phi}$ 是 Fokker–Planck 方程的平稳解（见[附录 B](../../appendices/b-density-evolution/index.md#app-continuity)）：

    $$
    \partial_t \rho = -\nabla \cdot(\rho  \nabla\log p_{\boldsymbol\phi}) + \tfrac{\sigma^2}{2}\Delta \rho.
    $$

    令 $\rho=p_{\boldsymbol\phi}$，得到 $(\tfrac{\sigma^2}{2}-1)\Delta p_{\boldsymbol\phi}=0$，这仅在 $\sigma=\sqrt{2}$ 时成立。

[^2]: Brownian 增量满足 $\mathbf{w}(t+\eta)-\mathbf{w}(t)\sim \mathcal{N}(\mathbf{0}, \eta \mathbf{I})$。因此，Euler–Maruyama 使用步进噪声 $\sqrt{2} [\mathbf{w}(t+\eta)-\mathbf{w}(t)] = \sqrt{2\eta} \boldsymbol{\epsilon}_n$，其中 $\boldsymbol{\epsilon}_n \sim \mathcal{N}(\mathbf{0},\mathbf{I})$；这解释了因子 $\sqrt{\eta}$，也是平方根缩放的来源。关于 Brownian 运动与 SDE 的详细介绍，请参阅[附录 A](../../appendices/a-differential-equations/index.md#app-de)。