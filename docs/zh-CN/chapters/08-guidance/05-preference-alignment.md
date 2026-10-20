# 8.5 从强化学习到用于模型对齐的直接偏好优化 {#sec-rlhf-dpo}

为了使生成模型与人类意图对齐，主流范式一直是基于人类反馈的强化学习（RLHF）。RLHF 虽然有效，却是一个复杂的多阶段过程，并可能不稳定。本节介绍*直接偏好优化（DPO）*（[Rafailov et al., 2023](../../references.md#ref-rafailov2023direct)），这是一种更简洁、稳定的方法，无需显式奖励建模或强化学习即可达到同一目标。随后介绍它通过 *Diffusion-DPO*（[Wallace et al., 2024](../../references.md#ref-wallace2024diffusion)）向扩散模型的扩展。

## 8.5.1 动机：绕过 RLHF 的困难 {#motivation-rlhf}

对齐的目标是将基础预训练模型（如 SFT 模型）引向人类偏好的输出。RLHF 分三个阶段。首先，*监督微调（SFT）*在提示词与回答的配对数据上训练基础模型。其次，*奖励建模（RM）*在偏好数据上拟合模型；数据包含提示词 $\mathbf{c}$ 及成对回答（受偏好的“胜者” $\mathbf{x}_w$ 与不受偏好的“败者” $\mathbf{x}_l$），学习标量 $r(\mathbf{c},\mathbf{x})$，使其满足 $r(\mathbf{c},\mathbf{x}_w)>r(\mathbf{c},\mathbf{x}_l)$。第三，*强化学习微调*使用 PPO 等算法（[Schulman et al., 2017](../../references.md#ref-schulman2017proximal)）优化 SFT 模型（策略 $\pi$[^1]），最大化来自 $r$ 的期望奖励，同时用 KL 惩罚进行正则化，使 $\pi$ 接近参考／SFT 分布。

尽管影响广泛，这一流程仍有缺点：强化学习阶段不稳定、计算开销高，因为它采用同策略方式，每次更新都需要当前模型新生成的样本；它还要求训练并运行多个大型模型（SFT、奖励模型，有时还包括价值模型）；此外，它优化的只是人类偏好的代理，因此奖励模型中的缺陷可能被利用。这引出了核心问题：

<span id="question-8-5-1"></span>

**问题 8.5.1**

能否去掉显式奖励建模和不稳定的强化学习步骤，直接在偏好数据上优化模型？

直接偏好优化（DPO）用一个类似监督学习的步骤替代多阶段 RLHF 流程，简化了对齐过程。DPO 不单独训练奖励模型，也不运行 PPO 等不稳定的强化学习算法，而是使用简单的逻辑损失，直接根据偏好对拟合策略，同时使其接近固定参考模型。关键认识是，KL 正则化 RLHF 目标可以改写，使策略与参考模型之间的对数似然比起到隐式奖励的作用。这保留了朝参考策略的同一种正则化，同时避免高成本的轨迹采样与显式奖励建模。

[第 8.5.2 节](05-preference-alignment.md#subsec-rlhf-bradley-terry)简要回顾 RLHF 流程及其对大型奖励模型和强化学习微调的依赖。[第 8.5.3 节](05-preference-alignment.md#subsec-dpo)介绍最初为语言模型提出的 DPO，它绕过奖励模型训练，简化对齐微调。最后，[第 8.5.4 节](05-preference-alignment.md#subsec-diffusion-dpo)将这一思想扩展到扩散模型，介绍作为生成建模中实用且稳定的对齐方法的 Diffusion-DPO。

## 8.5.2 RLHF：Bradley–Terry 视角 {#subsec-rlhf-bradley-terry}

**RLHF 简介。** RLHF 从学习到的评判器开始：奖励模型 $r_{\boldsymbol\psi}$ 为同一提示词 $\mathbf c$ 的候选回答赋予标量偏好评分。数据集 $\mathcal D$ 由样本对 $(\tilde{\mathbf x},\mathbf x)$ 组成，每对带有标签 $y$，表示是否更偏好 $\tilde{\mathbf x}$ 而非 $\mathbf x$。标签可以是二元值 $y\in\{0,1\}$，也可以是汇总多个评判者后得到的软标签 $y\in[0,1]$。训练目标是简单的逻辑损失

<span id="eq-rm-emp"></span>

$$
\begin{aligned}
    \mathcal L_{\mathrm{RM}}(\boldsymbol\psi)
=
- \mathbb E_{(\mathbf c,\tilde{\mathbf x},\mathbf x,y)\sim\mathcal D}
\Big[
y &\log \sigma \big(r_{\boldsymbol\psi}(\mathbf c,\tilde{\mathbf x})-r_{\boldsymbol\psi}(\mathbf c,\mathbf x)\big)
\\ &+(1-y)\log \big(1-\sigma \big(r_{\boldsymbol\psi}(\mathbf c,\tilde{\mathbf x})-r_{\boldsymbol\psi}(\mathbf c,\mathbf x)\big)\big)
\Big],
\end{aligned}
\tag{8.5.1}
$$

其中 $\sigma(u)=1/(1+e^{-u})$。实践中，$\mathcal D$ 中的偏好对可以来自多种来源：整理筛选的回答、不同检查点的模型快照，或预训练条件扩散模型的生成结果。常见约定是以有序格式 $(\text{winner}, \text{loser})$ 存储。按照这一约定，只需令 $y=1$，[式 (8.5.1)](05-preference-alignment.md#eq-rm-emp)就化为以下特例（取 $\tilde{\mathbf{x}} = {\mathbf{x}}^w$ 和 ${\mathbf{x}} = {\mathbf{x}}^l$）：

<span id="eq-rm-bt"></span>

$$
\mathcal L_{\mathrm{RM}}(\boldsymbol\psi)
=
- \mathbb E_{(\mathbf c,\mathbf x_w,\mathbf x_l)\sim\mathcal D}
\Big[\log \sigma \big(r_{\boldsymbol\psi}(\mathbf c,\mathbf x_w)-r_{\boldsymbol\psi}(\mathbf c,\mathbf x_l)\big)\Big].
\tag{8.5.2}
$$

**Bradley–Terry 视角及其与 KL 的联系。** 通常通过 Bradley–Terry（BT）模型（[Bradley and Terry, 1952](../../references.md#ref-bradley1952rank)）解释

$$
p_{r_{\boldsymbol\psi}}(\tilde{\mathbf x}\succ \mathbf x|\mathbf c)
:= \sigma \big(r_{\boldsymbol\psi}(\mathbf c,\tilde{\mathbf x})-r_{\boldsymbol\psi}(\mathbf c,\mathbf x)\big)
$$

该模型将两个标量评分转化为获胜概率。这一表述凸显了两个关键性质：(i) 只有评分的*差值*起作用（因此 $r_{\boldsymbol\psi}(\mathbf c,\cdot)$ 具有平移不变性）；(ii) 损失促使预测的胜者评分高于败者。为直观理解 (ii)，考虑一对标签为 $y\in\{0,1\}$ 的样本，并定义

$$
\Delta r := r_{\boldsymbol\psi}(\mathbf c,\tilde{\mathbf x})-r_{\boldsymbol\psi}(\mathbf c,\mathbf x),
\quad
p:=\sigma(\Delta r),\quad \sigma(u)=\tfrac{1}{1+e^{-u}}.
$$

单样本逻辑损失为

$$
\ell=-\big[y\log p+(1-y)\log(1-p)\big].
$$

于是

$$
\frac{\partial \ell}{\partial \Delta r}=\sigma(\Delta r)-y.
$$

在步长为 $\eta>0$ 的梯度下降下，评分差距按如下方式更新

$$
\Delta r \leftarrow \Delta r-\eta\big(\sigma(\Delta r)-y\big).
$$

因此，如果 $y=1$（“$\tilde{\mathbf x}$ 获胜”），则 $\sigma(\Delta r)-1\le 0$，所以 $\Delta r$ 增大（胜者上升、败者下降）；如果 $y=0$，则 $\Delta r$ 减小。

[式 (8.5.1)](05-preference-alignment.md#eq-rm-emp)中的每个单样本项，都可以看作观测 Bernoulli 标签与模型预测获胜概率之间的交叉熵：

$$
-\big[y\log p_{r_{\boldsymbol\psi}}+(1-y)\log(1-p_{r_{\boldsymbol\psi}})\big]
= \mathcal D_{\mathrm{KL}} \big(\mathrm{Bern}(y) \big\| \mathrm{Bern}(p_{r_{\boldsymbol\psi}})\big)
+ \mathcal H \big(\mathrm{Bern}(y)\big),
$$

其中，$\mathcal H$ 是目标 Bernoulli 分布的熵。在数据集 $\mathcal D$ 上平均，得到

<span id="eq-reward-loss"></span>

$$
\begin{aligned}
\mathcal L_{\mathrm{RM}}(\boldsymbol\psi)
= \mathbb E_{\mathcal D} \Big[\mathcal D_{\mathrm{KL}}\left(\mathrm{Bern}(y) \big\| \mathrm{Bern}(p_{r_{\boldsymbol\psi}})\right)\Big]
+ \underbrace{\mathbb E_{\mathcal D} \big[\mathcal H(\mathrm{Bern}(y))\big]}_{\text{independent of }\boldsymbol\psi}.
\end{aligned}
\tag{8.5.3}
$$

因此，最小化逻辑损失等价于最小化人类标签的经验 Bernoulli 分布与模型预测 Bernoulli 分布之间的 KL散度。在二元情形（$y\in\{0,1\}$）中，这一等价是精确的；对于软标签（$y\in[0,1]$），二者相差一个熵常数。直观地说，奖励模型通过训练调整获胜概率，直到它们与数据集中观测到的经验人类胜率一致。

从这里开始，我们采用最常见的约定，将 $\mathcal D$ 中的样本对以有序格式存储：$({\mathbf{x}}^w, {\mathbf{x}}^l, {\mathbf{c}}) \sim \mathcal D$。在这一约定下，标签始终为 $y=1$，损失简化为[式 (8.5.2)](05-preference-alignment.md#eq-rm-bt)给出的有序形式，后文将使用这一形式。

**KL 正则化策略优化（固定奖励）。** 给定通过[式 (8.5.2)](05-preference-alignment.md#eq-rm-bt)拟合的奖励 $r:=r_{\boldsymbol\psi^\times}$，以及条件预训练扩散模型 $p_{\boldsymbol{\phi}^\times}(\mathbf x|\mathbf c)$，RLHF 接着调整可学习策略 $\pi_{\boldsymbol\theta}(\mathbf x|\mathbf c)$，使其倾向于高奖励回答；该策略通常在 $p_{\boldsymbol{\phi}^\times}(\mathbf x|\mathbf c)$ 上微调。同时，使策略保持接近参考模型，参考模型取预训练扩散模型 $\pi_{\mathrm{ref}}(\mathbf x|\mathbf c):=p_{\boldsymbol{\phi}^\times}(\mathbf x|\mathbf c)$，并通过 $\mathcal{D}_{\mathrm{KL}}$ 惩罚进行正则化：

<span id="eq-rlhf-policy-correct"></span>

$$
\max_{\boldsymbol\theta}
\mathbb E_{\mathbf c\sim p(\mathbf c)}
\Big[
\mathbb E_{\mathbf x\sim \pi_{\boldsymbol\theta}(\cdot|\mathbf c)}\big[r_{\boldsymbol\psi}(\mathbf c,\mathbf x)\big]
-\beta \mathcal{D}_{\mathrm{KL}} \big(\pi_{\boldsymbol\theta}(\cdot|\mathbf c) \big\| \pi_{\mathrm{ref}}(\cdot|\mathbf c)\big)
\Big],
\tag{8.5.4}
$$

这明确体现了两种作用：寻找评判器偏好的样本，同时保持接近预训练参考模型。

[式 (8.5.2)](05-preference-alignment.md#eq-rm-bt)中的奖励目标只使用带标签样本对，并不要求 $\mathcal D$ 由参考模型（即预训练条件扩散模型）生成。虽然没有这一要求，但从接近目标策略的模型收集样本对，可以减少分布偏移，使学习到的奖励在实际使用区域更可靠。

因此，RLHF 分为两步：先通过最小化[式 (8.5.2)](05-preference-alignment.md#eq-rm-bt)中的损失来拟合奖励 $r^*$（等价于[式 (8.5.3)](05-preference-alignment.md#eq-reward-loss)中的期望二元 $\mathcal{D}_{\mathrm{KL}}$）；再通过求解[式 (8.5.4)](05-preference-alignment.md#eq-rlhf-policy-correct)优化策略 $\pi^*$。

## 8.5.3 DPO 框架 {#subsec-dpo}

**从 RLHF 建立联系。** 对于每个提示词 $\mathbf c$，给定拟合的奖励 $r := r_{\boldsymbol\psi^\times}$，[式 (8.5.4)](05-preference-alignment.md#eq-rlhf-policy-correct)中的 KL 正则化策略目标具有简单闭式解，可表达为如下基于能量的形式（[Peters et al., 2010](../../references.md#ref-peters2010relative)）：

<span id="eq-rlhf-closed-form"></span>

$$
\pi^*(\mathbf x|\mathbf c)
=
\frac{1}{Z(\mathbf c)} \pi_{\mathrm{ref}}(\mathbf x|\mathbf c)\exp(r(\mathbf c,\mathbf x)/\beta),
\tag{8.5.5}
$$

其中 $\pi_{\mathrm{ref}}(\mathbf x|\mathbf c):=p_{\boldsymbol{\phi}^\times}(\mathbf x|\mathbf c)$，且 $Z(\mathbf c)$ 是保证 $\int \pi^*(\mathbf x|\mathbf c) \mathop{}\!\mathrm{d} \mathbf x=1$ 的配分函数。

当 $\beta$ 较小时，$\exp(r/\beta)$ 变得更尖锐，使 $\pi^*$ 集中于高奖励区域：奖励占主导，策略进一步偏离 $\pi_{\mathrm{ref}}$，多样性下降，训练可能不稳定或容易出现奖励投机。当 $\beta$ 较大时，$\exp(r/\beta)$ 变平，使 $\pi^*$ 更接近 $\pi_{\mathrm{ref}}$：KL 项占主导，更新较保守，多样性接近参考模型，但奖励提升有限。

由于目标是直接微调策略（不单独训练奖励模型），[式 (8.5.5)](05-preference-alignment.md#eq-rlhf-closed-form)允许我们从任意策略*定义*一个*隐式奖励*。下面介绍这一做法。

**通过反解[式 (8.5.5)](05-preference-alignment.md#eq-rlhf-closed-form)定义隐式奖励。** [式 (8.5.5)](05-preference-alignment.md#eq-rlhf-closed-form)可以直接反解：对于任意策略 $\pi$（其支撑集包含于 $\pi_{\mathrm{ref}}$ 的支撑集），定义

<span id="eq-implicit-reward"></span>

$$
r_{\pi}(\mathbf c,\mathbf x)
 =
\beta \log\frac{\pi(\mathbf x|\mathbf c)}{\pi_{\mathrm{ref}}(\mathbf x|\mathbf c)}
 + \beta \log Z(\mathbf c).
\tag{8.5.6}
$$

那么，在[式 (8.5.5)](05-preference-alignment.md#eq-rlhf-closed-form)中用 $\pi$ 代替 $\pi^*$，该式仍成立；即 $\pi$ 是[式 (8.5.4)](05-preference-alignment.md#eq-rlhf-policy-correct)在奖励函数 $r_{\pi}$ 下的最优解。在这一意义上，$r_{\pi}$ 是*隐式（由策略诱导的）奖励*：它可以确定到一个依赖提示词的常数 $\beta\log Z(\mathbf c)$，而这一常数会在任何成对比较中消失，例如 BT 模型中的比较：

$$
r_{\pi}(\mathbf c,\mathbf x_w)-r_{\pi}(\mathbf c,\mathbf x_l)
=
\beta\Big(
\log\frac{\pi(\mathbf x_w|\mathbf c)}{\pi_{\mathrm{ref}}(\mathbf x_w|\mathbf c)}
-
\log\frac{\pi(\mathbf x_l|\mathbf c)}{\pi_{\mathrm{ref}}(\mathbf x_l|\mathbf c)}
\Big).
$$

正是这一抵消，使常数与偏好学习无关，并直接导出基于对数概率差的 DPO 损失。

**DPO 的训练损失。** 对于带标签样本对 $(\mathbf x_w,\mathbf x_l)$（二者对应同一提示词 $\mathbf c$），将隐式奖励[式 (8.5.6)](05-preference-alignment.md#eq-implicit-reward)代入[式 (8.5.2)](05-preference-alignment.md#eq-rm-bt)中的 BT 模型。胜者与败者之间的常数 $\log Z(\mathbf c)$ 抵消，得到一个基于对数概率差的逻辑损失目标：

$$
\begin{aligned}
\mathcal L_{\mathrm{DPO}}(\boldsymbol\theta;\pi_{\mathrm{ref}})
&=
-\mathbb E_{(\mathbf c,\mathbf x_w,\mathbf x_l)\sim\mathcal D} \left[
\log \sigma \Big(
\beta\big(
\log\frac{\pi_{\boldsymbol\theta}(\mathbf x_w|\mathbf c)}{\pi_{\mathrm{ref}}(\mathbf x_w|\mathbf c)}
-
\log\frac{\pi_{\boldsymbol\theta}(\mathbf x_l|\mathbf c)}{\pi_{\mathrm{ref}}(\mathbf x_l|\mathbf c)}
\big)
\Big)
\right].
\end{aligned}
$$

也就是说，DPO 提高胜者相对于败者的（经温度缩放的）优势，该优势由二者相对于参考模型的对数似然提升之差度量：

$$
-\log \sigma \left(
  \beta
  \bigl[
    \text{log-ratio difference of }
    \tfrac{\pi_{\boldsymbol\theta}}{\pi_{\mathrm{ref}}}
    \text{ at } \mathbf x_w \text{ vs. } \mathbf x_l
  \bigr]
\right).
$$

这在单个稳定的分类式微调阶段实现了 RLHF 的目标，无需训练显式奖励模型。

## 8.5.4 Diffusion-DPO {#subsec-diffusion-dpo}

本小节沿用[第 8.5.3 节](05-preference-alignment.md#subsec-dpo)的策略记号：$\pi_{\boldsymbol\theta}$ 和 $\pi_{\mathrm{ref}}$ 分别表示学习策略与参考扩散策略，包括它们的路径分布和转移分布；$p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$ 仍表示固定前向加噪过程。

**为什么直接套用 DPO 不适用于扩散模型？** 计算扩散模型中的样本似然 $\pi_{\boldsymbol{\theta}}(\mathbf x|\mathbf c)$，需要在求解 ODE 时使用瞬时变量替换公式（漂移的散度）（见[式 (4.3.7)](../04-score-sde-framework/03-training-and-sampling.md#eq-score-sde-likelihood)）[^2]，计算开销很大。此外，通过整个采样轨迹求导可能出现梯度消失或爆炸。为避免这些问题，Diffusion-DPO 在*路径*层面工作。我们以离散时间扩散模型（如 DDPM）为例，连续时间扩散模型类似。

**定义路径隐式奖励。** 设轨迹为 $\mathbf x_{0:T}:=(\mathbf x_T,\ldots,\mathbf x_0)$，它来自条件转移分布为 $\pi(\mathbf x_{t-1}|\mathbf x_t,\mathbf c)$ 的逆时间 Markov 链。其中，$\mathbf x_T$ 是噪声最高的状态，$\mathbf x_0$ 是干净输出。

人类偏好通过 $r(\mathbf c,\mathbf x_0)$ 定义在最终输出上。由于 $\mathbf x_0$ 是每条扩散轨迹的一部分，转到路径空间时不必改变奖励本身。有用的步骤是对整个逆向过程进行正则化。根据 KL散度的链式法则，完整路径分布之间的 KL散度，至少与其端点边缘分布之间的 KL散度一样大。这启发了路径目标

$$
\max_{{\boldsymbol{\theta}}}\mathbb E_{\mathbf c\sim p(\mathbf c)}
\Big[
\mathbb E_{\mathbf x_{0:T}\sim\pi_{{\boldsymbol{\theta}}}(\cdot|\mathbf c)}
\big[r(\mathbf c,\mathbf x_0)\big]
-\beta
\mathcal D_{\mathrm{KL}}
\big(
\pi_{{\boldsymbol{\theta}}}(\cdot|\mathbf c)
\big\|
\pi_{\mathrm{ref}}(\cdot|\mathbf c)
\big)
\Big],
$$

其中，$\pi_{{\boldsymbol{\theta}}}(\cdot|\mathbf c)$ 和 $\pi_{\mathrm{ref}}(\cdot|\mathbf c)$ 表示相应的路径分布。奖励项不变，而 KL 惩罚现在还控制潜在去噪轨迹偏离参考过程的程度。

在分布层面，假设归一化常数有限，则每个提示词 $\mathbf c$ 对应的最优解具有基于能量的形式

<span id="eq-path-closed-form"></span>

$$
\pi^*(\mathbf x_{0:T}|\mathbf c)
=
\frac{1}{Z(\mathbf c)}
\pi_{\mathrm{ref}}(\mathbf x_{0:T}|\mathbf c)
\exp\big(r(\mathbf c,\mathbf x_0)/\beta\big).
\tag{8.5.7}
$$

与普通 DPO 一样，可以反解这一关系，定义任意策略 $\pi$ 诱导的*隐式路径奖励*：

$$
R_{\pi}(\mathbf c,\mathbf x_{0:T})
:=
\beta\log
\frac{\pi(\mathbf x_{0:T}|\mathbf c)}
{\pi_{\mathrm{ref}}(\mathbf x_{0:T}|\mathbf c)}
+\beta\log Z(\mathbf c).
$$

在路径最优解处，$R_{\pi^*}(\mathbf c,\mathbf x_{0:T})=r(\mathbf c,\mathbf x_0)$。但对于一般策略，$R_{\pi}$ 还可能依赖潜在轨迹。由于偏好数据只比较干净端点，因此我们对与各端点关联的轨迹取平均，得到平均隐式路径奖励。

**从路径隐式奖励到 DPO。** 对于带标签端点对 $(\mathbf x_0^w,\mathbf x_0^l)$，在同一提示词 $\mathbf c$ 下，比较它们的隐式路径奖励的条件平均，可得到 Bradley–Terry 间隔：

<span id="eq-diffusion-dpo"></span>

$$
\begin{aligned}
\mathcal L_{\mathrm{Diff\text{-}DPO}}(\boldsymbol\theta;\pi_{\mathrm{ref}})
&:=
- \mathbb E_{(\mathbf c,\mathbf x_0^w,\mathbf x_0^l)\sim\mathcal D}
 \left[\log \sigma \big(\Delta R(\mathbf c;\boldsymbol\theta)\big)\right], \quad \text{where}\\
\Delta R(\mathbf c;\boldsymbol\theta)
&:=
\underbrace{\mathbb E_{\mathbf x_{1:T}^w\sim \pi_{\boldsymbol\theta}(\cdot | \mathbf x_0^w,\mathbf c)}
 \Big[R_{\pi_{\boldsymbol\theta}} \big(\mathbf c,(\mathbf x_0^w,\mathbf x_{1:T}^w)\big)\Big]}_{\text{winner path expectation}}
\\ &\quad\quad\quad-
\underbrace{\mathbb E_{\mathbf x_{1:T}^l\sim \pi_{\boldsymbol\theta}(\cdot | \mathbf x_0^l,\mathbf c)}
 \Big[R_{\pi_{\boldsymbol\theta}} \big(\mathbf c,(\mathbf x_0^l,\mathbf x_{1:T}^l)\big)\Big]}_{\text{loser path expectation}}.
\end{aligned}
\tag{8.5.8}
$$

这里，期望 $\mathbb E_{\mathbf x_{1:T}\sim\pi_{\boldsymbol\theta}(\cdot|\mathbf x_0,\mathbf c)}[\cdot]$ 的含义是：给定数据集中的固定端点 $\mathbf x_0$（如胜者 $\mathbf x_0^w$），对潜在去噪轨迹 $\mathbf x_{1:T}$ 取期望；轨迹来自模型诱导的条件路径分布（逆时间轨迹的后验），在转移核 $\pi_{\boldsymbol\theta}(\mathbf x_{t-1}|\mathbf x_t,\mathbf c)$ 下可能产生 $\mathbf x_0$。由于这些中间状态不可观测，我们在所有此类轨迹上平均路径奖励。

然而，[式 (8.5.8)](05-preference-alignment.md#eq-diffusion-dpo)由于三个实际原因而不可行：

1.  **端点条件化诱导难以计算的路径后验。** $\mathbb E_{\pi_{\boldsymbol\theta}(\mathbf x_{1:T}|\mathbf x_0,\mathbf c)}[\cdot]$ 对受约束、必须到达 $\mathbf x_0$ 的逆向路径取平均，而采样器运行 $\mathbf x_T \to \cdots \to \mathbf x_0$ 时没有这一约束。以端点为条件会产生扩散桥后验，一般没有闭式形式，采样开销也很大。

2.  **嵌套且与 $\boldsymbol\theta$ 耦合的期望。** 损失 $-\log\sigma(\Delta R(\mathbf c;\boldsymbol\theta))$，其中

    $$
    \Delta R=\mathbb E_{\text{paths}|\mathbf x_0^w,\mathbf c}[R_{\pi_{\boldsymbol\theta}}]
    -\mathbb E_{\text{paths}|\mathbf x_0^l,\mathbf c}[R_{\pi_{\boldsymbol\theta}}]
    $$

    其路径联合分布及被积函数 $R_{\pi_{\boldsymbol\theta}}$ 都依赖 $\boldsymbol\theta$。因此，$\nabla_{\boldsymbol\theta}$ 必须通过采样分布求导，产生 REINFORCE／路径导数耦合以及高方差梯度。

3.  **长链、大量求和与昂贵的反向传播。** 在 $R_{\pi_{\boldsymbol\theta}}(\mathbf c,\mathbf x_{0:T})$ 中，计算

    $$
    \beta \left[\log \pi_{\boldsymbol\theta}(\mathbf x_{0:T}|\mathbf c)
    -\log \pi_{\mathrm{ref}}(\mathbf x_{0:T}|\mathbf c)\right]
    $$

    需要 $\mathcal O(T)$ 个逐步对数密度，其中 $T \sim 10^2$–$10^3$；策略 $\pi_{\boldsymbol\theta}$ 和参考 $\pi_{\mathrm{ref}}$ 都需要计算，胜者与败者路径也都需要计算。通过这些随机链（或桥采样器）反向传播，会消耗大量内存和计算资源，并可能不稳定；若每对样本使用多条轨迹，并对所有三元组重复进行，训练就会超出实际预算。

**为[式 (8.5.8)](05-preference-alignment.md#eq-diffusion-dpo)构造可计算的代理目标。** 为使其可计算，我们利用一个关键数学认识：结合扩散模型性质并应用 Jensen 不等式，可以优化这一损失的可计算上界。这将问题从计算整条路径的似然，转化为计算路径内各单步转移上的期望：

由于 $-\log\sigma(\cdot)$ 是凸函数，Jensen 不等式允许将内部期望移到非线性函数外，从而给出上界：

$$
\begin{aligned}
\mathcal L_{\mathrm{Diff\text{-}DPO}}(\boldsymbol\theta;\pi_{\mathrm{ref}})
\le
-\mathbb E_{(\mathbf c,\mathbf x_0^w,\mathbf x_0^l)\sim\mathcal D}
  \mathbb E_{\substack{\mathbf x^w_{1:T}\sim \pi_{\boldsymbol\theta}(\cdot | \mathbf x_0^w,\mathbf c)\\[1pt]
                       \mathbf x^l_{1:T}\sim \pi_{\boldsymbol\theta}(\cdot | \mathbf x_0^l,\mathbf c)}}
\left[
\log \sigma \Big(
R_{\pi_{\boldsymbol\theta}}(\mathbf c,\mathbf x^w_{0:T})
-
R_{\pi_{\boldsymbol\theta}}(\mathbf c,\mathbf x^l_{0:T})
\Big)
\right].
\end{aligned}
$$

利用隐式奖励恒等式 $R_{\pi_{\boldsymbol\theta}}=\beta\log\frac{\pi_{\boldsymbol\theta}}{\pi_{\mathrm{ref}}}+\beta\log Z(\mathbf c)$，以及胜者与败者之间的常数抵消，上界变为

<span id="eq-dpo-path-bound"></span>

$$
\begin{aligned}
\mathcal L_{\mathrm{Diff\text{-}DPO}}(\boldsymbol\theta;\pi_{\mathrm{ref}})
& \le
-\mathbb E
\left[
\log \sigma \Big(
\beta\big(
\log\frac{\pi_{\boldsymbol\theta}(\mathbf x^w_{0:T}| \mathbf c)}{\pi_{\mathrm{ref}}(\mathbf x^w_{0:T}| \mathbf c)}
-
\log\frac{\pi_{\boldsymbol\theta}(\mathbf x^l_{0:T}| \mathbf c)}{\pi_{\mathrm{ref}}(\mathbf x^l_{0:T}| \mathbf c)}
\big)
\Big)
\right].
\end{aligned}
\tag{8.5.9}
$$

**可计算的代理目标（逐步形式）。** 现在利用逆向过程的 Markov 性，分解 $\mathcal L_{\mathrm{Diff\text{-}DPO}}$ 的上界。这使我们能够将路径层面的偏好表示为逐步贡献之和，将难以计算的路径损失转化为可计算的单步估计器。得到的形式是 DPO 式的对数 sigmoid 损失，其内部间隔化为 DSM 式的 MSE 之差。具体来说，对于逆向链，

$$
\begin{aligned}
\pi_{\boldsymbol\theta}(\mathbf x_{0:T}| \mathbf c)
&=\pi_{\boldsymbol\theta}(\mathbf x_T| \mathbf c) \prod_{t=1}^{T}
\pi_{\boldsymbol\theta}(\mathbf x_{t-1}| \mathbf x_t,\mathbf c),
\\
\pi_{\mathrm{ref}}(\mathbf x_{0:T}| \mathbf c)
&=\pi_{\mathrm{ref}}(\mathbf x_T| \mathbf c) \prod_{t=1}^{T}
\pi_{\mathrm{ref}}(\mathbf x_{t-1}| \mathbf x_t,\mathbf c).
\end{aligned}
$$

因此

$$
\frac{\pi_{\boldsymbol\theta}(\mathbf x_{0:T}| \mathbf c)}{\pi_{\mathrm{ref}}(\mathbf x_{0:T}| \mathbf c)}
=
\frac{\pi_{\boldsymbol\theta}(\mathbf x_T| \mathbf c)}{\pi_{\mathrm{ref}}(\mathbf x_T| \mathbf c)}
\prod_{t=1}^{T}
\frac{\pi_{\boldsymbol\theta}(\mathbf x_{t-1}| \mathbf x_t,\mathbf c)}
     {\pi_{\mathrm{ref}}(\mathbf x_{t-1}| \mathbf x_t,\mathbf c)}.
$$

如果两个模型在时刻 $T$ 的先验相同，即 $\pi_{\boldsymbol\theta}(\mathbf x_T| \mathbf c)=\pi_{\mathrm{ref}}(\mathbf x_T| \mathbf c)$，那么第一个因子等于 $1$，取对数得到

$$
\log\frac{\pi_{\boldsymbol\theta}(\mathbf x_{0:T}| \mathbf c)}{\pi_{\mathrm{ref}}(\mathbf x_{0:T}| \mathbf c)}
=\sum_{t=1}^{T}
\log\frac{\pi_{\boldsymbol\theta}(\mathbf x_{t-1}| \mathbf x_t,\mathbf c)}
         {\pi_{\mathrm{ref}}(\mathbf x_{t-1}| \mathbf x_t,\mathbf c)}.
$$

由此，[式 (8.5.9)](05-preference-alignment.md#eq-dpo-path-bound)中的上界可以写为

$$
\mathcal L_{\mathrm{Diff\text{-}DPO}}(\boldsymbol\theta;\pi_{\mathrm{ref}})
  \le  - {\mathbb{E}} \left[\log\sigma \Big(\beta \sum_{t=1}^T \Delta_t\Big)\right],
$$

其中，每个时间步的贡献为

$$
\Delta_t
= \log\frac{\pi_{\boldsymbol\theta}(\mathbf x^w_{t-1}|\mathbf x^w_t,\mathbf c)}
               {\pi_{\mathrm{ref}}(\mathbf x^w_{t-1}|\mathbf x^w_t,\mathbf c)}
- \log\frac{\pi_{\boldsymbol\theta}(\mathbf x^l_{t-1}|\mathbf x^l_t,\mathbf c)}
               {\pi_{\mathrm{ref}}(\mathbf x^l_{t-1}|\mathbf x^l_t,\mathbf c)}.
$$

为将时间步求和化为单个随机选择的转移，注意 $\beta\sum_{s=1}^T\Delta_s = {\mathbb{E}}_{t\sim\mathcal U\{1,\dots,T\}}[\beta T\Delta_t]$。由于 $-\log\sigma(\cdot)$ 为凸函数，在取已有外层期望之后，Jensen 不等式给出

$$
\mathcal L_{\mathrm{Diff\text{-}DPO}}(\boldsymbol\theta;\pi_{\mathrm{ref}})
\leq
-{\mathbb{E}}\left[
\log\sigma\bigl(\beta T\Delta_t\bigr)
\right],
$$

其中，期望针对 $(\mathbf c,\mathbf x_0^w,\mathbf x_0^l)\sim\mathcal D$、$t\sim\mathcal U\{1,\dots,T\}$，以及 $\mathbf x_{t-1:t}^w\sim \pi_{\boldsymbol\theta}(\cdot|\mathbf x_0^w,\mathbf c)$ 和 $\mathbf x_{t-1:t}^l\sim \pi_{\boldsymbol\theta}(\cdot|\mathbf x_0^l,\mathbf c)$。虽然上界现在只依赖一个转移，但对这些以端点为条件的转移对进行采样，仍然难以计算。

按照 Diffusion-DPO 的做法，我们用固定前向加噪过程近似难以计算的模型诱导条件路径。对于每个干净端点 $\mathbf x_0$，先采样 $\mathbf x_t \sim p_t(\mathbf x_t|\mathbf x_0)$。给定 $(\mathbf x_0,\mathbf x_t)$，前向过程诱导可计算的后验 $p(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)$。在该后验下对逆向转移对数比取平均，得到

$$
\begin{aligned}
&{\mathbb{E}}_{p(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)}
\left[
\log
\frac{
\pi_{\boldsymbol\theta}(\mathbf x_{t-1}|\mathbf x_t,\mathbf c)
}{
\pi_{\mathrm{ref}}(\mathbf x_{t-1}|\mathbf x_t,\mathbf c)
}
\right]
\\&=
-
\mathcal D_{\mathrm{KL}}
\left(
p(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)
\,\middle\|\,
\pi_{\boldsymbol\theta}(\mathbf x_{t-1}|\mathbf x_t,\mathbf c)
\right)+
\mathcal D_{\mathrm{KL}}
\left(
p(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)
\,\middle\|\,
\pi_{\mathrm{ref}}(\mathbf x_{t-1}|\mathbf x_t,\mathbf c)
\right).
\end{aligned}
$$

对于具有相同给定逆向方差的 Gaussian 逆向核，这些 KL散度化为加权去噪回归误差。在 $\boldsymbol\epsilon$ 预测下，

<span id="eq-dpo-gaussian-log-ratio"></span>

$$
\begin{aligned}
{\mathbb{E}}_{p(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)}
\left[
\log
\frac{
\pi_{\boldsymbol\theta}(\mathbf x_{t-1}|\mathbf x_t,\mathbf c)
}{
\pi_{\mathrm{ref}}(\mathbf x_{t-1}|\mathbf x_t,\mathbf c)
}
\right]
=
-\lambda_t
\left[
\left\|
\hat{\boldsymbol\epsilon}_{\boldsymbol\theta}(\mathbf x_t,t,\mathbf c)
-\boldsymbol\epsilon
\right\|^2
-
\left\|
\hat{\boldsymbol\epsilon}_{\mathrm{ref}}(\mathbf x_t,t,\mathbf c)
-\boldsymbol\epsilon
\right\|^2
\right],
\end{aligned}
\tag{8.5.10}
$$

其中 $\mathbf x_t = \alpha_t\mathbf x_0 + \sigma_t\boldsymbol\epsilon$，且 $\boldsymbol\epsilon\sim\mathcal N(\mathbf 0,\mathbf I)$；$\lambda_t>0$ 汇集了依赖调度的权重因子。

为简化记号，对于任意加噪样本 $(\mathbf x_t,\boldsymbol\epsilon)$，定义：

$$
\Delta\mathrm{MSE}(\mathbf x_t;\boldsymbol\epsilon)
:= \big\|\hat{\boldsymbol\epsilon}_{\boldsymbol\theta}(\mathbf x_t,t,\mathbf c)-\boldsymbol\epsilon\big\|^2
 - \big\|\hat{\boldsymbol\epsilon}_{\mathrm{ref}}(\mathbf x_t,t,\mathbf c)-\boldsymbol\epsilon\big\|^2.
$$

由此得到针对 $\mathcal L_{\mathrm{Diff\text{-}DPO}}$ 的实用单步 Diffusion-DPO 代理目标：

$$
\begin{aligned}
&\tilde{\mathcal L}_{\mathrm{Diff\text{-}DPO}}(\boldsymbol\theta;\pi_{\mathrm{ref}})
    :=
    \\&-\mathbb E_{\substack{(\mathbf c,\mathbf x_0^w,\mathbf x_0^l)\sim\mathcal D\\
    t\sim\mathcal U\{1,\dots,T\}, \boldsymbol\epsilon^w,\boldsymbol\epsilon^l\sim\mathcal N(\mathbf 0,\mathbf I)}}
    \Big[
    \log\sigma \Big(
    \beta T\lambda_t\big(\Delta\mathrm{MSE}(\mathbf x_t^l;\boldsymbol\epsilon^l)-\Delta\mathrm{MSE}(\mathbf x_t^w;\boldsymbol\epsilon^w)\big)
    \Big)
    \Big],
\end{aligned}
$$

其中 $\mathbf x_t^w=\alpha_t\mathbf x_0^w+\sigma_t\boldsymbol\epsilon^w$ 且 $\mathbf x_t^l=\alpha_t\mathbf x_0^l+\sigma_t\boldsymbol\epsilon^l$。实践中，也可以使用共享噪声 $\boldsymbol\epsilon^w=\boldsymbol\epsilon^l$ 作为降低方差的变体；这不改变目标整体的 DPO 式结构，但会稍微改变随机估计器。

直观而言，最小化 $\tilde{\mathcal L}_{\mathrm{Diff\text{-}DPO}}$ 会增大相对于参考模型归一化后的胜者对败者的间隔。等价地，它鼓励模型相对于冻结的参考模型，在胜者上比在败者上更多地降低去噪误差。参考模型不直接贡献梯度项，但仍位于 $\log\sigma(\cdot)$ 非线性内部的间隔中，因此依然影响 DPO 式的逐样本权重。

[^1]: 策略将提示词／历史（状态）映射为回答／动作上的分布。

[^2]: 在离散时间扩散模型（如 DDPM）中，计算 $\pi_{\boldsymbol\theta}(\mathbf x_0| \mathbf c)$ 需要对潜在逆向轨迹 $\mathbf x_{1:T}$ 进行边缘化。
