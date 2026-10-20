# 4.3 Score SDE 的训练与采样 {#sec-scoresde-training-sampling}

## 4.3.1 训练 {#subsec-sde-train}

沿用[第 3 章](../03-score-based-perspective/index.md#ch-score-based)的思路，我们使用以时间为条件的神经网络来近似 oracle score $\nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}})$：

$$
\mathbf{s}_{\boldsymbol{\phi}} = \mathbf{s}_{\boldsymbol{\phi}}({\mathbf{x}}, t)
$$

这种近似覆盖所有 $t\in[0,T]$，通过最小化[式 (3.2.1)](../03-score-based-perspective/02-score-based-models.md#eq-sm)那样的 score 匹配目标实现：

$$
\mathcal{L}_{\text{SM}}(\boldsymbol{\phi}; \omega(\cdot)) := \frac{1}{2} \mathbb{E}_{t\sim p_{\text{time}}}  \mathbb{E}_{{\mathbf{x}}_t \sim p_t}\left[\omega(t) \left\|\mathbf{s}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t) - \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t)\right\|_2^2\right],
$$

其中 $p_{\text{time}}$ 是某种时间分布（例如 $[0,T]$ 上的均匀分布），$\omega(\cdot)$ 是时间权重函数。

<span id="fig-score-and-denoising-score"></span>

![去噪 score 匹配中的边缘 score 与条件 score](../../../en/assets/chapter-04/denoising-score-matching.png)

**图 4.6：** **去噪 score 匹配技巧。** **（左）** 对于固定的含噪样本 ${\mathbf{x}}_t$，边缘 score $\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t)$ 指向含噪样本更可能出现的区域。直接计算它，需要综合所有可能的干净样本 ${\mathbf{x}}_0$ 的贡献，这些样本都可能产生 ${\mathbf{x}}_t$，因此通常不可计算。**（右）** 然而，训练时干净样本 ${\mathbf{x}}_0$ 是已知的，它被用于生成 ${\mathbf{x}}_t$。对于 Gaussian 扰动，相应条件 score $\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$ 具有闭式表达式，为回归提供一个简单向量目标。在训练样本上，这些条件 score 向量平均后得到边缘 score。因此，去噪 score 匹配可以学得所需的边缘 score，无须直接对通常不可计算的 score $\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t)$ 进行回归，后者的求值需要对数据分布中所有可能样本取平均。

来源：作者借助 AI 辅助编程绘制。

为避免依赖不可计算的 oracle score $\nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}})$，我们使用[式 (3.3.2)](../03-score-based-perspective/03-denoising-score-matching.md#eq-dsm-loss)中的 DSM 损失。以数据点 ${\mathbf{x}}_0$ 为条件，可借助[式 (D.2.4)](../../appendices/d-supplementary-proofs/02-score-based-perspective.md#eq-score-gaussian-formula)使用解析可计算的 score $\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$，具体示例见[第 4.4 节](04-sde-instantiations.md#sec-instances)。具体而言，使用以下损失函数：

<span id="eq-conti-dsm-loss-real"></span>

$$
\begin{aligned}
     &\mathcal{L}_{\text{DSM}}(\boldsymbol{\phi}; \omega(\cdot)) := \\
     &\frac{1}{2} \mathbb{E}_{t}  \mathbb{E}_{{\mathbf{x}}_0} \mathbb{E}_{ p_t({\mathbf{x}}_t|{\mathbf{x}}_0)} \Big[\omega(t)\left\lVert\mathbf{s}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t) - \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t|{\mathbf{x}}_0)\right\rVert_2^2 \Big],
\end{aligned}
\tag{4.3.1}
$$

其中 ${\mathbf{x}}_0 \sim p_{\mathrm{data}}$。[式 (4.3.1)](03-training-and-sampling.md#eq-conti-dsm-loss-real)可理解为[式 (3.4.1)](../03-score-based-perspective/04-noise-conditional-score-networks.md#eq-nscn)的连续时间对应形式，其中离散情形的求和被替换为积分。

与[定理 3.3.1](../03-score-based-perspective/03-denoising-score-matching.md#thm-sm-dsm)中的结果类似，[式 (4.3.1)](03-training-and-sampling.md#eq-conti-dsm-loss-real)的最优解可唯一确定如下：

<span id="dsm-minimizer"></span>

**命题 4.3.1：DSM 的最优解**

最优解 $\mathbf{s}^*$ 满足

<span id="eq-optimal-score"></span>

$$
\begin{aligned}
    \mathbf{s}^*({\mathbf{x}}_t, t)
    = \mathbb{E}_{{\mathbf{x}}_0 \sim p({\mathbf{x}}_0 | {\mathbf{x}}_t)} \big[ \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t | {\mathbf{x}}_0) \big]= \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t),
\end{aligned}
\tag{4.3.2}
$$

对几乎处处的 ${\mathbf{x}}_t \sim p_t$ 与 $t \in [0,T]$ 成立。

**命题证明。**

DSM 目标可理解为一个最小二乘误差问题。具体而言，在每个时刻 $t$，最优 score 函数是对数条件密度梯度的条件期望，根据 Bayes 法则，它等于对数边缘密度的梯度。详细证明见[第 D.2.6 节](../../appendices/d-supplementary-proofs/02-score-based-perspective.md#app-sec-min-dsm)。

## 4.3.2 采样与推断 {#subsec-sde-sample}

<span id="fig-sde-ode-trajectories"></span>

![随机与确定性生成轨迹](../../../en/assets/chapter-04/sampling-sde-ode-trajectories.png)

**图 4.7：** **从 Score SDE 采样的二维示意图。** 前向 SDE 设计为 $\mathop{}\!\mathrm{d}{\mathbf{x}}(t)=\sqrt{2t}\mathop{}\!\mathrm{d}{\mathbf{w}}(t)$，漂移为 $\mathbf{f}\equiv\mathbf{0}$，扩散系数为 $g(t)=\sqrt{2t}$，定义在 $[0,T]$ 上，将双峰 Gaussian 混合分布 $p_0=p_{\mathrm{data}}$ 演化为 $p_T\approx p_{\mathrm{prior}}:=\mathcal{N}(\mathbf{0},T^2\mathbf{I})$。图中展示逆时间 SDE（蓝色，通过[式 (4.3.4)](03-training-and-sampling.md#eq-euler-maruyama)求解）与 PF-ODE（红色，通过[式 (4.3.6)](03-training-and-sampling.md#eq-euler-ode)求解）的采样轨迹。从随机点 ${\mathbf{x}}_T\sim p_{\mathrm{prior}}$（深色“$\times$”）出发，两条轨迹都移向 $p_{\mathrm{data}}$ 的支撑集，这发生在 $t\to 0$ 时。

来源：作者绘制。

学得

$$
\mathbf{s}_{\boldsymbol{\phi}^\times} := \mathbf{s}_{\boldsymbol{\phi}^\times}({\mathbf{x}}, t) \approx \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}),
$$

后，我们将不可计算的 oracle score $\nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}})$ 从逆时间 SDE（[式 (4.1.6)](01-score-sde-principles.md#eq-sde-backward)）与 PF-ODE（[式 (4.1.7)](01-score-sde-principles.md#eq-prob-ode-gt)）中替换为学得的近似 $\mathbf{s}_{\boldsymbol{\phi}^\times}({\mathbf{x}}, t)$。这使我们能够通过 SDE 或 ODE 进行可计算的推断。为清楚起见，将所得过程分别记为 ${\mathbf{x}}^{\mathrm{SDE}}_{\boldsymbol{\phi}^\times}(t)$ 与 ${\mathbf{x}}^{\mathrm{ODE}}_{\boldsymbol{\phi}^\times}(t)$，但后文将省略这一区分。[^1]

**经验逆时间 SDE。** 将训练好的 score 模型 $\mathbf{s}_{\boldsymbol{\phi}^\times}$ 代入[式 (4.1.6)](01-score-sde-principles.md#eq-sde-backward)，替换真实 score，即得到用于生成的参数化逆时间 SDE：

<span id="eq-sde-backward-sub"></span>

$$
\mathop{}\!\mathrm{d} {\mathbf{x}}^{\mathrm{SDE}}_{\boldsymbol{\phi}^\times}(t) = \left[\mathbf{f}\left({\mathbf{x}}^{\mathrm{SDE}}_{\boldsymbol{\phi}^\times}(t), t\right) - g^2(t) \mathbf{s}_{\boldsymbol{\phi}^\times}\left({\mathbf{x}}^{\mathrm{SDE}}_{\boldsymbol{\phi}^\times}(t), t\right) \right] \mathop{}\!\mathrm{d} t + g(t) \mathop{}\!\mathrm{d} \bar{{\mathbf{w}}}(t).
\tag{4.3.3}
$$

为了生成样本，先抽取初值 ${\mathbf{x}}_T$，其来自先验分布 $p_{\mathrm{prior}}$，再沿时间反向从 $t=T$ 到 $t=0$ 数值求解[式 (4.3.3)](03-training-and-sampling.md#eq-sde-backward-sub)。一种标准数值求解器是 Euler–Maruyama 方法，其离散更新规则为：

<span id="eq-euler-maruyama"></span>

$$
{\mathbf{x}}_{t - \Delta t} \leftarrow {\mathbf{x}}_t - \left[ \mathbf{f}({\mathbf{x}}_t, t) - g^2(t)  \mathbf{s}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t, t) \right] \Delta t + g(t) \sqrt{\Delta t} \cdot \boldsymbol{\epsilon},
\tag{4.3.4}
$$

其中 $\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \mathbf{I})$，$\Delta t > 0$ 为步长。

反复执行这一更新，得到最终样本 ${\mathbf{x}}^{\mathrm{SDE}}_{\boldsymbol{\phi}^\times}(0)$。若 score 模型准确，这些生成样本的分布记为 $p_{\boldsymbol{\phi}^\times}^{\mathrm{SDE}}(\cdot; 0)$，将很好地近似真实数据分布[^2]：

$$
p_{\boldsymbol{\phi}^\times}^{\mathrm{SDE}}(\cdot; 0) \approx p_{\mathrm{data}}(\cdot).
$$

事实上，对于 VP 构造，[式 (2.2.14)](../02-variational-perspective/02-ddpm.md#eq-ddpm-sampling)中的 DDPM 采样方案，与上述 Euler–Maruyama 离散化具有相同的连续时间逆向 SDE 极限；见[第 4.4 节](04-sde-instantiations.md#sec-instances)。

**经验 PF-ODE。** PF-ODE 定义了连接 $p_{\mathrm{prior}}$ 与 $p_{\mathrm{data}}$ 的连续流，可用于采样、编码以及精确的似然计算。下文将分别介绍这些操作。

**I. 使用 PF-ODE 采样。** 用 $\mathbf{s}_{\boldsymbol{\phi}^\times}$ 替换[式 (4.1.7)](01-score-sde-principles.md#eq-prob-ode-gt)中的 oracle score，得到经验 PF-ODE：

<span id="eq-prob-ode-sub"></span>

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} {\mathbf{x}}^{\mathrm{ODE}}_{\boldsymbol{\phi}^\times}(t) = \mathbf{f}\left({\mathbf{x}}^{\mathrm{ODE}}_{\boldsymbol{\phi}^\times}(t), t\right) - \frac{1}{2} g^2(t) \mathbf{s}_{\boldsymbol{\phi}^\times}\left({\mathbf{x}}^{\mathrm{ODE}}_{\boldsymbol{\phi}^\times}(t), t\right).
\tag{4.3.5}
$$

为了生成样本，先抽取初始样本 ${\mathbf{x}}_T$，其来自先验分布 $p_{\mathrm{prior}}$。随后沿时间反向，从 $t=T$ 到 $t=0$ 数值求解[式 (4.3.5)](03-training-and-sampling.md#eq-prob-ode-sub)中的 PF-ODE。这等价于近似以下积分：

$$
{\mathbf{x}}^{\mathrm{ODE}}_{\boldsymbol{\phi}^\times}(0) = {\mathbf{x}}_T + \int_T^0 \left[ \mathbf{f}\left({\mathbf{x}}^{\mathrm{ODE}}_{\boldsymbol{\phi}^\times}(\tau), \tau\right) - \frac{1}{2} g^2(\tau) \mathbf{s}_{\boldsymbol{\phi}^\times}\left({\mathbf{x}}^{\mathrm{ODE}}_{\boldsymbol{\phi}^\times}(\tau), \tau\right) \right] \mathrm{d}\tau.
$$

求解该积分得到最终样本 ${\mathbf{x}}^{\mathrm{ODE}}_{\boldsymbol{\phi}^\times}(0)$。通过这一确定性过程生成的样本分布记为 $p_{\boldsymbol{\phi}^\times}^{\mathrm{ODE}}(\cdot; 0)$，它近似数据分布，即 $p_{\boldsymbol{\phi}^\times}^{\mathrm{ODE}}(\cdot; 0) \approx p_{\mathrm{data}}$。

令 $\Delta t > 0$ 表示离散化步长。一种标准数值积分方法是 Euler 方法，它估计

$$
\mathbf{f}({\mathbf{x}}_\tau, \tau) - \frac{1}{2} g^2(\tau)
\mathbf{s}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau, \tau) \approx \mathbf{f}({\mathbf{x}}_t, t) - \frac{1}{2} g^2(t)
\mathbf{s}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t, t), \quad \tau \in [t-\Delta t, t],
$$

从而得到以下更新规则：

<span id="eq-euler-ode"></span>

$$
{\mathbf{x}}_{t - \Delta t} \leftarrow {\mathbf{x}}_t -
\left[ \mathbf{f}({\mathbf{x}}_t, t) - \frac{1}{2} g^2(t)
\mathbf{s}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t, t) \right] \Delta t.
\tag{4.3.6}
$$

这一联系使我们能够通过以下核心认识重新表述生成过程：

<span id="insight-4-3-1"></span>

**启示 4.3.1：生成 $\Leftrightarrow$ 求解 ODE/SDE**

从扩散模型中采样，本质上等价于求解对应的概率流 ODE 或逆时间 SDE。

这一等价关系清楚地解释了[问题 3.5.1](../03-score-based-perspective/05-comparison.md#ques-dm-sampling-slow)所提出的扩散模型采样缓慢问题。生成的计算成本高，是因为这些微分方程的数值求解器本质上需要迭代，通常需要很多步才能准确近似解轨迹[^3]。不过，PF-ODE 形式也具有优势，因为它使我们能够使用关于加速数值求解器的大量现有研究。[第 9 章](../09-numerical-solvers/index.md#ch-numerical-solvers)将重点探索如何利用这些技术加速扩散模型采样。

**II. 使用 PF-ODE 反演。** 如前所述，与 SDE 不同，同一个[式 (4.3.5)](03-training-and-sampling.md#eq-prob-ode-sub)既可以沿时间正向求解（从 $0$ 到 $T$），也可以反向求解（从 $T$ 到 $0$）。正向求解时，ODE 流在所有 $t \in [0,T]$ 上将数据映射为其含噪潜在表示，起到编码器的作用。这一概念带来了可控生成中的强大应用，例如图像转换、编辑等（[Mokady et al., 2023](../../references.md#ref-mokady2023null)；[Su et al., 2022](../../references.md#ref-su2022dual)）。

**III. 通过 PF-ODE 计算对数似然。** PF-ODE 还允许计算数据样本的似然。回顾[式 (5.1.9)](../05-flow-based-perspective/01-normalizing-flows-and-neural-odes.md#eq-instant-change-of-var)，对于 ODE，沿轨迹的对数密度变化由其速度场的散度决定。

对于学得的 PF-ODE，定义

$$
{\mathbf{v}}_{\boldsymbol{\phi}^\times}({\mathbf{x}},t)
:=
{\mathbf{f}}({\mathbf{x}},t)
-\frac{1}{2}g^2(t){\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}},t).
$$

若 ${\mathbf{x}}(t)$ 按以下方式演化：

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}{\mathbf{x}}(t)
=
{\mathbf{v}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}(t),t),
$$

则其对数密度满足

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}
\log p^{\text{ODE}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}(t);t)
=
-\nabla_{{\mathbf{x}}}\cdot
{\mathbf{v}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}(t),t).
$$

为了计算观测样本 ${\mathbf{x}}_0$ 的似然，我们从 $t=0$ 到 $t=T$ 正向运行 PF-ODE，将 ${\mathbf{x}}_0$ 映射到一个终点，其中模型密度为可计算的先验。还可以通过以下定义同时跟踪对数密度变化：

$$
\delta(t)
:=
\log p^{\text{ODE}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}(t);t)
-
\log p^{\text{ODE}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_0;0).
$$

于是，$\delta(0)=0$，并且状态与对数密度变化可通过下式一起计算：

<span id="eq-score-sde-likelihood"></span>

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}
\begin{bmatrix}
{\mathbf{x}}(t)\\
\delta(t)
\end{bmatrix}
=
\begin{bmatrix}
{\mathbf{v}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}(t),t)\\
-\nabla_{{\mathbf{x}}}\cdot
{\mathbf{v}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}(t),t)
\end{bmatrix},
\qquad
\begin{bmatrix}
{\mathbf{x}}(0)\\
\delta(0)
\end{bmatrix}
=
\begin{bmatrix}
{\mathbf{x}}_0\\
0
\end{bmatrix}.
\end{aligned}
\tag{4.3.7}
$$

积分到 $t=T$ 后，终端密度为 $p_{\mathrm{prior}}({\mathbf{x}}(T))$。因此，

$$
\log
p^{\text{ODE}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_0;0)
=
\log p_{\mathrm{prior}}({\mathbf{x}}(T))
-
\delta(T).
$$

等价地，

$$
\log
p^{\text{ODE}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_0;0)
=
\log p_{\mathrm{prior}}({\mathbf{x}}(T))
+
\int_0^T
\nabla_{{\mathbf{x}}}\cdot
{\mathbf{v}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}(t),t)
\,\mathop{}\!\mathrm{d} t.
$$

因此，用于生成与反演的同一个 PF-ODE，也提供了用于计算似然的连续变量替换公式。

[^1]: 这样做是为了简化本小节之后的记号。

[^2]: 理论上，估计精度取决于 $p_T$ 与 $p_{\mathrm{prior}}$ 之间的差异（通常可忽略）、模型训练误差以及数值离散化误差（[De Bortoli, 2022](../../references.md#ref-de2022convergence)；[Wang et al., 2024](../../references.md#ref-wang2024evaluating)）。此处不讨论形式化的界。

[^3]: 例如，DDPM 与 Score SDE 在生成时通常使用 $1,000$ 次函数求值。