# 4.1 Score SDE 的原理 {#sec-dm-today}

<span id="fig-time-dependent-score"></span>

![跨噪声水平随时间变化的密度与 score 景观](../../../en/assets/chapter-04/time-dependent-score.png)

**图 4.1：** **随时间变化的 score 景观示意图。** 随着噪声水平增加，扰动密度 $p_t({\mathbf{x}})$ 从集中在数据模式附近的多模态分布（数据），演化为宽广、近乎单峰的分布（先验）。相应地，score 场 $\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}})$ 从将样本拉向邻近模式的强烈局部向量场，变为指向整体质心的平滑全局场。这个随时间变化的 score 场，正是基于 score 的扩散模型（即 Score SDE）学习的关键对象。

来源：作者借助 AI 辅助编程绘制。

使用多个噪声尺度是 NCSN 与 DDPM 框架取得成功的关键因素。本节介绍 *Score SDE*（[Song et al., 2020](../../references.md#ref-song2020score)）的基础，它通过考虑连续的噪声水平，进一步发展这一思想。[Sohl-Dickstein et al. (2015)](../../references.md#ref-sohl2015deep) 已经注意到前向与逆向扩散过程的连续时间极限，而 [Song et al. (2020)](../../references.md#ref-song2020score) 将数据演化表述为随机／常微分方程，使这一视角成为核心，其中噪声水平随时间平滑增加。这一形式的核心是*随时间变化的 score 函数* $\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}})$：在每个噪声水平 $t$ 上，它指向当前含噪分布 $p_t$ 中密度较高的区域，而恰恰是这个量决定如何逆转扩散。如[图 4.1](01-score-sde-principles.md#fig-time-dependent-score)所示，score 场从 $t$ 较小时强烈汇聚到各个数据模式，演化为 $t$ 较大时平缓地指向全局中心。连续时间形式不仅统一了先前的离散时间模型，还为生成建模提供了有理论依据且灵活的基础：学习所有噪声水平上的这一族 score 函数，就将生成归结为求解微分方程。

## 4.1.1 动机：从离散过程到连续时间过程 {#subsec-scoresde-motivation}

我们重新考察 NCSN 与 DDPM 的前向噪声注入方案。NCSN 使用一系列递增的噪声水平 $\{\sigma_i\}_{i=1}^L$。每个干净样本 ${\mathbf{x}}_0 \sim p_{\mathrm{data}}$ 被扰动为

$$
{\mathbf{x}}_{\sigma_i} = {\mathbf{x}}_0 + \sigma_i \boldsymbol{\epsilon}_i,
\qquad \boldsymbol{\epsilon}_i \sim \mathcal{N}(\boldsymbol{0}, {\mathbf{I}}).
$$

DDPM 则按照方差调度 $\{\beta_i\}_{i=1}^L$ 逐步注入噪声[^1]：

$$
{\mathbf{x}}_i = \sqrt{1 - \beta_i}\, {\mathbf{x}}_{i-1} + \sqrt{\beta_i}\, \boldsymbol{\epsilon}_i,
\qquad \boldsymbol{\epsilon}_i \sim \mathcal{N}(\boldsymbol{0}, {\mathbf{I}}).
$$

我们在离散时间网格上统一考察二者，其中从 ${\mathbf{x}}_t$ 到 ${\mathbf{x}}_{t+\Delta t}$ 的顺序更新形式为[^2]：

$$
\begin{aligned}
\text{\textbf{NCSN:}} \,\,
    & {\mathbf{x}}_{t+\Delta t} = {\mathbf{x}}_t + \sqrt{\sigma_{t+\Delta t}^2 - \sigma_t^2} \boldsymbol{\epsilon}_t
    &&\approx {\mathbf{x}}_t + \sqrt{\frac{\mathop{}\!\mathrm{d} \sigma_t^2}{\mathop{}\!\mathrm{d} t} \Delta t}  \boldsymbol{\epsilon}_t \\[0.5em]
    \text{\textbf{DDPM:}} \,\,
    & {\mathbf{x}}_{t+\Delta t} = \sqrt{1 - \beta(t)\Delta t} {\mathbf{x}}_t + \sqrt{\beta(t)\Delta t} \boldsymbol{\epsilon}_t
    &&\approx {\mathbf{x}}_t - \frac{1}{2} \beta(t) {\mathbf{x}}_t  \Delta t + \sqrt{\beta(t)\Delta t}  \boldsymbol{\epsilon}_t,
\end{aligned}
$$

其中 $\boldsymbol{\epsilon}_t \sim \mathcal{N}(\boldsymbol{0}, {\mathbf{I}})$。有趣的是，两种噪声注入过程都遵循共同的结构：

<span id="eq-euler-discrete-sde"></span>

$$
\begin{aligned}
{\mathbf{x}}_{t+\Delta t} \approx {\mathbf{x}}_t + {\mathbf{f}}({\mathbf{x}}_t, t) \Delta t + g(t) \sqrt{\Delta t}\, \boldsymbol{\epsilon}_t,
\end{aligned}
\tag{4.1.1}
$$

其中 ${\mathbf{f}}: \mathbb{R}^D \times \mathbb{R} \to \mathbb{R}^D$ 与 $g: \mathbb{R} \to \mathbb{R}$ 为：

$$
\begin{aligned}
\text{\textbf{NCSN:}} \quad
    & {\mathbf{f}}({\mathbf{x}}, t) = 0, \qquad g(t) = \sqrt{\frac{\mathop{}\!\mathrm{d} \sigma^2(t)}{\mathop{}\!\mathrm{d} t}} \\[0.5em]
    \text{\textbf{DDPM:}} \quad
    & {\mathbf{f}}({\mathbf{x}}, t) = -\frac{1}{2} \beta(t)\, {\mathbf{x}}, \qquad g(t) = \sqrt{\beta(t)}.
\end{aligned}
$$

<span id="fig-sde-forward-noise"></span>

![离散噪声注入收敛到连续时间前向 SDE](../../../en/assets/chapter-04/sde-forward-noise.png)

**图 4.2：** **离散时间加噪步骤示意图。** 它从 $t$ 到 $t + \Delta t$ 添加噪声，均值漂移为 ${\mathbf{f}}({\mathbf{x}}_t, t)$，扩散系数为 $g(t)$。

来源：作者绘制。

这一形式对应以下 Gaussian 转移：

<span id="eq-transition-forward-f-g"></span>

$$
\begin{aligned}
p({\mathbf{x}}_{t+\Delta t} |{\mathbf{x}}_t) := \mathcal{N}\!\left({\mathbf{x}}_{t+\Delta t}; {\mathbf{x}}_t + {\mathbf{f}}({\mathbf{x}}_t, t)\Delta t,\, g^2(t)\Delta t\, {\mathbf{I}}\right),
\end{aligned}
\tag{4.1.2}
$$

其中略微宽泛地使用记号，将 ${\mathbf{x}}_t$ 视为固定样本，将 ${\mathbf{x}}_{t+\Delta t}$ 视为随机变量。

当 $\Delta t \to 0$ 时（概念上可理解为准备*无穷多*层噪声），离散时间过程收敛到沿时间前向演化的连续时间 SDE[^3]：

$$
\begin{aligned}
\mathop{}\!\mathrm{d} {\mathbf{x}}(t) = {\mathbf{f}}({\mathbf{x}}(t), t) \mathop{}\!\mathrm{d} t + g(t) \mathop{}\!\mathrm{d} {\mathbf{w}}(t),
\end{aligned}
$$

其中 ${\mathbf{w}}(t)$ 是标准 Wiener 过程（或 Brownian 运动）。

**说明。**

此处无须完整的形式化定义，但可以将 *Wiener 过程*理解为连续时间随机过程 ${\mathbf{w}}(t)$：它从零开始，具有独立增量，且对任意 $s < t$，增量 ${\mathbf{w}}(t) - {\mathbf{w}}(s)$ 服从均值为零、方差为 $t - s$ 的正态分布。它表示独立 Gaussian 涨落随时间的累积；虽然几乎必然连续，却处处不可微。

在无穷小时间间隔 $[t, t + \mathop{}\!\mathrm{d} t]$ 上，Wiener 过程的增量定义为

$$
\mathop{}\!\mathrm{d} {\mathbf{w}}(t) := {\mathbf{w}}(t + \mathop{}\!\mathrm{d} t) - {\mathbf{w}}(t),
$$

将其建模为均值为零、方差为 $\mathop{}\!\mathrm{d} t$ 的 Gaussian 随机变量：

$$
\mathop{}\!\mathrm{d} {\mathbf{w}}(t) \sim \mathcal{N}(\boldsymbol{0}, \mathop{}\!\mathrm{d} t {\mathbf{I}}).
$$

[第 A.2 节](../../appendices/a-differential-equations/02-stochastic-differential-equations.md#app-sde-intro)简要介绍 SDE 的基础，[附录 C](../../appendices/c-ito-calculus/index.md#app-ito)则作更深入的讨论。不过，在概念上，我们可以按以下方式理解离散与连续形式之间的联系：

-   ${\mathbf{x}}(t+\Delta t) - {\mathbf{x}}(t) \approx \mathop{}\!\mathrm{d} {\mathbf{x}}(t)$,

-   $\Delta t \approx \mathop{}\!\mathrm{d} t$,

-   $\sqrt{\Delta t} \boldsymbol{\epsilon}_t \sim \mathcal{N}(\boldsymbol{0}, \Delta t {\mathbf{I}}) \approx \mathop{}\!\mathrm{d} {\mathbf{w}}(t)$.

一旦指定漂移 ${\mathbf{f}}({\mathbf{x}},t)$ 与扩散 $g(t)$，前向 SDE 就自动诱导出一个逆时间 SDE，将终端噪声分布输运回数据分布。逆向动力学只包含一个未知项，令人意外的是，它恰好就是*每个连续时间水平上的 score 函数*。这使 score 匹配成为训练目标；学得 score 后，采样就转化为利用学得的 score 对逆时间 SDE 进行数值积分。

[第 4.3 节](03-training-and-sampling.md#sec-scoresde-training-sampling)将介绍实际实现；在此之前，我们先在[第 4.1.2 节](01-score-sde-principles.md#subsec-forward-sde)与[第 4.1.3 节](01-score-sde-principles.md#subsec-reverse-sde)考察前向和逆向过程的理论基础。

## 4.1.2 前向 SDE：从数据到噪声 {#subsec-forward-sde}

<span id="fig-sde-density-traj"></span>

![前向过程的密度演化与样本轨迹](../../../en/assets/chapter-04/forward-sde-ode-trajectories.png)

**图 4.3：** **扩散模型前向过程的一维可视化。** 过程从采样的初始点（标记为“$\times$”）开始，这些点来自复杂双峰数据分布 $p_0 = p_{\mathrm{data}}$，向简单的单峰 Gaussian 先验 $p_T \approx p_{\mathrm{prior}}$ 演化。背景热图展示随时间逐渐平滑的边缘概率密度 $p_t$。图中展示了从 $t=0$ 到 $t=T$ 演化的样本轨迹，对比随机前向 SDE 过程（蓝色路径）与其确定性对应过程 PF-ODE（白色路径）。注意，PF-ODE 是密度的确定性输运映射，通常并不等于从单点出发的样本路径的均值。

来源：作者绘制。

通过这种形式，NCSN（[Song and Ermon, 2019](../../references.md#ref-song2019generative)）与 DDPM（[Sohl-Dickstein et al., 2015](../../references.md#ref-sohl2015deep)；[Ho et al., 2020](../../references.md#ref-ho2020denoising)）等早期离散时间方法，可以统一到*连续时间*框架中。该框架中的随机过程 ${\mathbf{x}}(t)$ 由区间 $[0, T]$ 上定义的前向 SDE 支配：

<span id="eq-sde-forward"></span>

$$
\mathop{}\!\mathrm{d}{\mathbf{x}}(t) = \mathbf{f}({\mathbf{x}}(t), t) \mathop{}\!\mathrm{d} t + g(t) \mathop{}\!\mathrm{d}{\mathbf{w}}(t), \quad {\mathbf{x}}(0) \sim p_{\mathrm{data}}.
\tag{4.1.3}
$$

这里，$\mathbf{f}(\cdot, t)\colon\mathbb{R}^D\to\mathbb{R}^D$ 是漂移，$g(t) \in \mathbb{R}$ 是标量扩散系数，${\mathbf{w}}(t)$ 表示标准 Wiener 过程。我们将其称为*前向 SDE*，它描述干净数据如何随时间逐渐受到扰动而变成噪声。

一旦指定漂移 $\mathbf{f}$ 与扩散系数 $g$，前向过程就被完全确定，描述数据变量如何通过注入 Gaussian 噪声而逐步受到扰动。具体而言，它诱导出两族随时间变化的密度：

**扰动核。** 条件分布

$$
p_t({\mathbf{x}}_t |{\mathbf{x}}_0)
$$

描述干净数据样本 ${\mathbf{x}}_0 \sim p_{\mathrm{data}}$ 如何演化为相应的含噪样本 ${\mathbf{x}}_t$，对应时刻 $t$。一般而言，[式 (4.1.3)](01-score-sde-principles.md#eq-sde-forward)中的漂移项 ${\mathbf{f}}({\mathbf{x}},t)$ 可以是 ${\mathbf{x}}$ 的任意函数，但常见且便于解析处理的选择，是假设它为仿射形式：

<span id="eq-affine-drift"></span>

$$
{\mathbf{f}}({\mathbf{x}},t) = f(t) {\mathbf{x}},
\tag{4.1.4}
$$

其中 $f(t)$ 是 $t$ 的标量函数，通常取非正值。在这一结构下，给定 ${\mathbf{x}}_0$ 后，条件过程保持 Gaussian 形式，扰动核也可通过求解相应的均值与方差 ODE 得到闭式解（[Särkkä and Solin, 2019](../../references.md#ref-sarkka2019applied)）（另见[第 4.4.3 节](04-sde-instantiations.md#subsec-scoresde-perturbation-kernel)）。

具体而言，

$$
p_t({\mathbf{x}}_t |{\mathbf{x}}_0) = \mathcal{N} \big({\mathbf{x}}_t; {\mathbf{m}}(t), P(t)\mathbf I_D\big),
$$

其中

$$
{\mathbf{m}}(t) = \exp \Big(\int_0^t f(u) \mathop{}\!\mathrm{d} u\Big) {\mathbf{x}}_0,
\quad
P(t) = \int_0^t \exp \Big(2 \int_s^t f(u) \mathop{}\!\mathrm{d} u\Big) g^2(s) \mathop{}\!\mathrm{d} s,
$$

初始条件为 ${\mathbf{m}}(0)={\mathbf{x}}_0$、$P(0)=0$。

这一显式形式使我们能够直接采样 ${\mathbf{x}}_t$，条件为 ${\mathbf{x}}_0$，而无须数值积分 SDE，因此称为*无须模拟*。NCSN 与 DDPM 都属于这种仿射漂移设定。

下文将针对任意漂移 ${\mathbf{f}}({\mathbf{x}},t)$ 发展一般理论，但在闭式分析有帮助时，会重新回到仿射漂移。

**边缘密度。** 时间边缘密度 $p_t({\mathbf{x}}_t)$ 通过对扰动核积分得到：

<span id="eq-scoresde-oracle-marginal"></span>

$$
p_t({\mathbf{x}}_t) := \int p_t({\mathbf{x}}_t|{\mathbf{x}}_0) p_{\mathrm{data}}({\mathbf{x}}_0) \mathop{}\!\mathrm{d} {\mathbf{x}}_0, \quad \text{with } p_0 = p_{\mathrm{data}}.
\tag{4.1.5}
$$

通过适当选择系数 $f(t)$ 与 $g(t)$，前向过程逐渐添加噪声，直到初始状态的影响实际上被遗忘。当 $T$ 变大时，条件分布 $p_T(\mathbf{x}_T |\mathbf{x}_0)$ 不再依赖于 $\mathbf{x}_0$，因为它的均值按以下方式演化：

$$
{\mathbf{m}}(T) = \exp\!\Big(\int_0^T f(u)\mathop{}\!\mathrm{d} u\Big) \mathbf{x}_0  \longrightarrow  \mathbf{0}, \quad \text{as } T\to\infty,
$$

这里要求 $f(u)$ 非正，使指数因子衰减。同时，方差增长并稳定下来，以匹配选定的先验分布。因此，边缘分布

$$
p_T(\mathbf{x}_T) = \int p_T(\mathbf{x}_T |\mathbf{x}_0) p_{\mathrm{data}}(\mathbf{x}_0)\mathop{}\!\mathrm{d}\mathbf{x}_0,
$$

最初表示数据样本上的复杂混合分布，随后收敛到简单先验 $p_{\mathrm{prior}}$，通常为 Gaussian 分布。在这一极限下，

$$
p_T(\mathbf{x}_T) \approx p_{\mathrm{prior}}(\mathbf{x}_T)
\quad \text{and} \quad
p_T(\mathbf{x}_T |\mathbf{x}_0) \approx p_{\mathrm{prior}}(\mathbf{x}_T),
$$

于是，前向过程将任意数据分布映射为可计算的先验，为逆转和生成提供便利的起点。

## 4.1.3 用于生成的逆时间随机过程 {#subsec-reverse-sde}

<span id="fig-sde-density-back-traj"></span>

![逆时间 SDE 轨迹从噪声演化为数据](../../../en/assets/chapter-04/reverse-sde-trajectories.png)

**图 4.4：** **用于数据生成的逆时间随机过程可视化。** 它从简单先验分布 $p_{\mathrm{prior}}$ 中抽取的样本出发，起始时刻为 $t=T$（标记为“$\times$”），并使用逆向 SDE 沿时间反向演化。所得轨迹在 $t=0$ 结束，共同形成目标双峰数据分布 $p_0 = p_{\mathrm{data}}$。背景热图展示概率密度如何从简单 Gaussian 分布逐渐变为复杂目标分布。

来源：作者绘制。

直观上，从噪声生成数据可通过“逆转”前向过程实现：从先验分布随机采样一个点，使其沿时间反向演化，即可得到生成样本。对于确定性系统（即 ODE），这一思想自然成立。由于不涉及随机性，逆转时间仅意味着沿前向过程的同一路径反向追踪一个点的轨迹[^4]。相比之下，SDE 在每个时间步都包含随机性，这意味着一个点可以沿许多可能的随机轨迹演化。因此，逆转此类过程更为微妙[^5]。

虽然单条随机轨迹不可逆，一个关键发现是，这些轨迹上的分布可以逆转。[Anderson (1982)](../../references.md#ref-anderson1982reverse) 的基础性结果将其形式化：对于[式 (4.1.3)](01-score-sde-principles.md#eq-sde-forward)中的前向过程，其时间反演过程 $\{ \bar{{\mathbf{x}}}(t) \}_{t \in [0, T]}$[^6] 本身也由一个定义良好的 SDE 支配。该逆时间过程从 $T$ 演化到 $0$，动力学为：

<span id="eq-sde-backward"></span>

$$
\begin{aligned}[t]
    \mathop{}\!\mathrm{d}\bar{{\mathbf{x}}}(t) = \left[\mathbf{f}(\bar{{\mathbf{x}}}(t), t) {\color{orange} - g^2(t) \nabla_{{\mathbf{x}}} \log p_t(\bar{{\mathbf{x}}}(t))} \right] \mathop{}\!\mathrm{d} t + g(t) \mathop{}\!\mathrm{d} \bar{{\mathbf{w}}}(t), \\
    \bar{{\mathbf{x}}}(T) \sim p_{\mathrm{prior}} \approx p_T.
\end{aligned}
\tag{4.1.6}
$$

这里，$\bar{{\mathbf{w}}}(t)$ 表示逆时间中的标准 Wiener 过程，定义为 $\bar{{\mathbf{w}}}(t) := {\mathbf{w}}(T - t) - {\mathbf{w}}(T)$。

**说明。**

本章始终使用 $t$ 表示原始扩散时钟：$t=0$ 对应数据，$t=T$ 对应噪声。因此，[式 (4.1.6)](01-score-sde-principles.md#eq-sde-backward)中的逆时间 SDE 从 $t=T$ 积分到 $t=0$，沿逆向过程有 $\mathop{}\!\mathrm{d} t<0$。等价地，也可以引入正向运行的逆向时钟 $s:=T-t$；除非另行说明，我们保留原始时钟 $t$。

为建立对[式 (4.1.6)](01-score-sde-principles.md#eq-sde-backward)的直觉，我们在[第 4.2.2 节](02-matching-marginals.md#subsec-example-gaussian)给出一个具体示例，其中数据分布为 Gaussian，动力学为线性 Gaussian。该设定可以解析处理：仅使用基础微积分与线性代数，就能直接推导时间反演公式，无须调用 [Anderson (1982)](../../references.md#ref-anderson1982reverse) 的完整一般理论。

注意，随机性 $g\neq 0$ 的存在引入额外修正项 $-g^2(t) \nabla_{{\mathbf{x}}} \log p_t(\bar{{\mathbf{x}}}(t))$，它计入扩散的影响，保证逆向动力学正确重现前向 SDE 诱导的边缘分布演化（见[第 4.2 节](02-matching-marginals.md#subsec-fpe-sde-ode)）。

**从概念上看，逆向过程为何有效？** [第 4.6.2 节](06-fokker-planck-and-reverse-sde.md#subsec-reverse-sde-reason)通过将逆时间 SDE 与 DDPM 变分框架联系起来，给出直观推导（属于选读内容，但颇有启发）。这里补充说明逆时间动力学如何从噪声恢复有结构的数据。

乍看之下，逆时间过程中出现 Brownian 噪声似乎有些矛盾。如果前向扩散把数据散布到越来越嘈杂的配置中，那么逆转这一过程，尤其是通过 $\bar{{\mathbf{w}}}(t)$ 继续引入额外随机性，为何能得到集中在数据流形附近、干净且有结构的样本？关键在于，逆时间 SDE 注入的并非任意随机性。扩散项 $g(t)\mathop{}\!\mathrm{d}\bar{{\mathbf{w}}}(t)$ 始终与 score 驱动的漂移 $- g^2(t)\nabla_{{\mathbf{x}}}\log p_t(\bar{{\mathbf{x}}}(t))$ 耦合。二者相互平衡：score 引导轨迹移向更高密度区域，噪声则引入受控的随机性，使过程能够探索，又不至于让随机性淹没动力学。

为更清楚地理解这一点，回到[式 (3.1.6)](../03-score-based-perspective/01-energy-based-models.md#eq-continuous-langevin)的 Langevin 直觉。当 $f(t)\equiv 0$ 时，[式 (4.1.6)](01-score-sde-principles.md#eq-sde-backward)写为

$$
\mathop{}\!\mathrm{d}\bar{{\mathbf{x}}}(t)
= - g^2(t) \nabla_{{\mathbf{x}}}\log p_t\!\big(\bar{{\mathbf{x}}}(t)\big) \mathop{}\!\mathrm{d} t
+ g(t)  \mathop{}\!\mathrm{d}\bar{{\mathbf{w}}}(t).
$$

通过 $s:=T-t$ 正向重参数化时间（因此 $\mathop{}\!\mathrm{d} t=-\mathop{}\!\mathrm{d} s$），并在分布意义下重新记 Brownian 运动，使 $\mathop{}\!\mathrm{d}\bar{{\mathbf{w}}}(t)=-\mathop{}\!\mathrm{d}{\mathbf{w}}_s$。写作 $\bar{{\mathbf{x}}}_s:=\bar{{\mathbf{x}}}(T-s)$ 与 $\pi_s:=p_{T-s}$，则得到

$$
\begin{aligned}
\mathop{}\!\mathrm{d}\bar{{\mathbf{x}}}_s
&= g^2(T-s) \nabla\log \pi_s \big(\bar{{\mathbf{x}}}_s\big) \mathop{}\!\mathrm{d} s
+ g(T-s) \mathop{}\!\mathrm{d}{\mathbf{w}}_s
\\&=
2\tau(s) \nabla\log \pi_s\!\big(\bar{{\mathbf{x}}}_s\big) \mathop{}\!\mathrm{d} s
+ \sqrt{2\tau(s)} \mathop{}\!\mathrm{d}{\mathbf{w}}_s,
\quad
\tau(s):=\tfrac{1}{2}g^2(T-s).
\end{aligned}
$$

这具有温度随时间变化的 Langevin 形式，温度为 $\tau(s)$，目标为演化中的密度 $\pi_s$。由 Tweedie 公式（[式 (3.3.6)](../03-score-based-perspective/03-denoising-score-matching.md#eq-tweedie)），score 方向 $\nabla\log \pi_s$ 在每个时间切片上都指向条件干净信号，因此漂移不断“拉回”去噪后的结构。

关键在于，$g(t)$ 沿逆向轨迹进行*退火*。初期（$s\approx 0$，即 $t\approx T$），$g(T-s)$ 通常较大，因此注入的噪声较强，过程进行广泛探索。随着 $s$ 增加，$g(T-s)$ 减小，随机项减弱，score 项占据主导，将样本拉入 $\pi_s$ 的高密度区域；到 $s=T$（即 $t=0$）时，轨迹集中在数据流形附近。

**逆时间 SDE 能力概览。**

随时间变化的 score 函数

$$
{\mathbf{s}}({\mathbf{x}}, t) := \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}})
$$

自然地出现在[式 (4.1.6)](01-score-sde-principles.md#eq-sde-backward)中，这一点十分引人关注。一旦指定前向系数 $f(t)$ 与 $g(t)$，score 就成为逆向动力学中唯一剩余的未知量。这凸显了它的核心作用：有了 score，逆向过程就被确定，采样只需使用学得的 score 对[式 (4.1.6)](01-score-sde-principles.md#eq-sde-backward)进行数值积分。

由于 oracle score 通常没有闭式表达式，我们采用[第 3 章](../03-score-based-perspective/index.md#ch-score-based)的方法，训练神经网络 $\mathbf{s}_{\boldsymbol{\phi}}({\mathbf{x}}, t)$，通过 score 匹配近似它；详见[第 4.3.1 节](03-training-and-sampling.md#subsec-sde-train)。在[式 (4.1.6)](01-score-sde-principles.md#eq-sde-backward)中将 ${\mathbf{s}}({\mathbf{x}}, t)$ 替换为 $\mathbf{s}_{\boldsymbol{\phi}}({\mathbf{x}}, t)$，就能完全确定逆向动力学。

生成对应于反向求解逆时间 SDE：从 $t = T$ 开始，以 ${\mathbf{x}}_T \sim p_{\mathrm{prior}}$ 为初值，求解到 $t = 0$。[Anderson (1982)](../../references.md#ref-anderson1982reverse) 证明，前向与逆向过程的边缘密度一致，从而保证 $t=0$ 时的样本近似服从 $p_{\mathrm{data}}$，条件是 $p_{\mathrm{prior}} \approx p_T$。我们将在[第 4.3.2 节](03-training-and-sampling.md#subsec-sde-sample)进一步讨论。

## 4.1.4 用于生成的确定性过程（概率流 ODE） {#subsec-pf-ode}

虽然[式 (4.1.6)](01-score-sde-principles.md#eq-sde-backward)中的 SDE 引入随机性，可能提高生成样本的多样性，但随之产生一个问题：

<span id="question-4-1-1"></span>

**问题 4.1.1**

是否必须使用[式 (4.1.6)](01-score-sde-principles.md#eq-sde-backward)中的 SDE 进行采样？

受 [Maoutsa et al. (2020)](../../references.md#ref-maoutsa2020interacting) 启发，[Song et al. (2020)](../../references.md#ref-song2020score) 还引入了一个确定性过程，即一个 ODE，使样本演化具有与前向 SDE 相同的边缘分布。这个过程 $\{\tilde{\mathbf{x}}(t)\}_{t\in[0,T]}$[^7] 称为*概率流 ODE（PF-ODE）*，形式为：

<span id="eq-prob-ode-gt"></span>

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}\tilde{\mathbf{x}}(t) = \mathbf{f}(\tilde{\mathbf{x}}(t), t) - \frac{1}{2} g^2(t) \nabla_{{\mathbf{x}}} \log p_t(\tilde{\mathbf{x}}(t)).
\tag{4.1.7}
$$

与 SDE 情形类似，可以用学得的近似替换真实 score，并从 $t = T$ 到 $t = 0$ 对逆时间 ODE 积分以生成样本。具体而言，生成样本（PF-ODE 在时刻 $t=0$ 的解）形式为

$$
\tilde{\mathbf{x}}(T) + \int_{T}^{0} \Big[ \mathbf{f}(\tilde{\mathbf{x}}(\tau), \tau) - \frac{1}{2} g^2(\tau)  \nabla_{{\mathbf{x}}} \log p_\tau(\tilde{\mathbf{x}}(\tau)) \Big]   \mathop{}\!\mathrm{d} \tau,
$$

其中初始条件为 $\tilde{\mathbf{x}}(T) \sim p_{\mathrm{prior}}$。由于该积分无法以闭式形式计算，实际生成依赖数值求解器（例如 Euler 方法，见[式 (4.3.4)](03-training-and-sampling.md#eq-euler-maruyama)）。

与逆时间 SDE 相比，PF-ODE 具有两个关键优势：

-   ODE 可以沿两个方向积分，从 $t=0$ 到 $t=T$，或从 $t=T$ 到 $t=0$，使用相同的方程形式，只需在所选端点指定对应初始条件。这种双向性与通常只允许沿时间正向积分的 SDE 不同。

-   它可以使用大量成熟、现成的 ODE 数值求解器。

我们强调，PF-ODE 并非通过简单去除[式 (4.1.6)](01-score-sde-principles.md#eq-sde-backward)中的扩散项得到；尤其是，其漂移项中的因子 $\frac{1}{2}$ 具有理论依据。总体而言，[式 (4.1.7)](01-score-sde-principles.md#eq-prob-ode-gt)通过恰当选择 ODE 的漂移，使其演化保持与[式 (4.1.3)](01-score-sde-principles.md#eq-sde-forward)的前向 SDE 相同的边缘密度。下一节将详细介绍保证这些边缘分布一致的基本原理，即 Fokker–Planck 方程（[Øksendal, 2003](../../references.md#ref-oksendal2003stochastic)）。

[^1]: 在[第 2.2 节](../02-variational-perspective/02-ddpm.md#sec-ddpm)中，DDPM 前向步骤写为 ${\mathbf{x}}_i = \sqrt{1-\beta_i^2}\,{\mathbf{x}}_{i-1} + \beta_i\,\boldsymbol{\epsilon}_i$，其中 $\beta_i$ 表示注入噪声的标准差，且 $\alpha_i := \sqrt{1-\beta_i^2}$。这里以及本章其余部分遵循 [Song et al. (2020)](../../references.md#ref-song2020score) 的约定，$\beta_i$ 表示注入噪声的方差，因此 ${\mathbf{x}}_i = \sqrt{1-\beta_i}\,{\mathbf{x}}_{i-1} + \sqrt{\beta_i}\,\boldsymbol{\epsilon}_i$。两种约定通过 $\beta_i^{\text{(here)}} = \bigl(\beta_i^{\text{(Ch. 2)}}\bigr)^2$ 联系，得到完全相同的动力学。

[^2]: 为方便起见，我们交替使用 ${\mathbf{x}}(t)$ 与 ${\mathbf{x}}_t$（其他随时间变化的变量也类似）表示时刻 $t$ 的样本。

[^3]: 当 $\Delta t\to0$ 时，[式 (4.1.2)](01-score-sde-principles.md#eq-transition-forward-f-g)中的前向核收敛到相应 Itô SDE 的解。完全严格的证明依赖较深入的结果，此处请参阅相关文献。

[^4]: 从技术上说，这对应于通过时间翻转代换 $t \leftrightarrow T - t$ 求解 ODE。

[^5]: 直接翻转时间并不能得到正确的逆向过程。

[^6]: 我们使用上横线记号，将逆向过程 $\{ \bar{{\mathbf{x}}}(t) \}_{t \in [0, T]}$ 与前向 SDE 定义的前向过程 $\{ {\mathbf{x}}(t) \}_{t \in [0, T]}$ 区分开来。

[^7]: 我们使用波浪号，将这一过程与前向及逆时间 SDE 对应的过程区分开来。下文为简洁起见省略这一区分。