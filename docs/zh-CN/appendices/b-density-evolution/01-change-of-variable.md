# B.1 变量变换公式：从确定性映射到随机流 {#sec-cov}

本节通过与微积分中的经典变量变换公式类比，解释连续性方程和 Fokker–Planck 方程。我们从熟悉的单变量情形出发，扩展到多变量及概率密度（[第 B.1.1 节](01-change-of-variable.md#subsec-cov-det)），再推广到双射映射的复合，其连续时间极限给出连续性方程（[第 B.1.2 节](01-change-of-variable.md#subsec-cov-continuity-eq)）。最后，通过加入随机噪声纳入随机性，自然将连续性方程扩展为 Fokker–Planck 方程（[第 B.1.3 节](01-change-of-variable.md#subsec-cov-fpe)）。

## B.1.1 确定性映射的变量变换公式 {#subsec-cov-det}

我们按确定性映射移动粒子，研究其*分布*（密度）如何演化。关键原则是概率质量守恒，建立在微积分与概率论的一个基本结果之上：变量变换公式。该公式描述积分，以及由此决定的概率密度，如何在光滑双射映射下变换。为建立直观认识，先考虑单次更新，再扩展到连续的一系列变换。

**单次更新。** 考虑将向量场（类似力）$\boldsymbol{\Psi}:{\mathbb{R}}^D \to {\mathbb{R}}^D$ 作用一个单位时间所诱导的单次更新规则。从初始粒子状态 ${\mathbf{x}}_0$ 出发，下一状态为

$$
{\mathbf{x}}_1 = \boldsymbol{\Psi}({\mathbf{x}}_0).
$$

**底层模式（密度）及其移动方式。** 若初始状态遵循由密度 $p_0$ 描述的底层“规律／模式”（即 ${\mathbf{x}}_0\sim p_0$），则应用 $\boldsymbol{\Psi}$ 会产生新密度 $p_1$，对应的状态为 ${\mathbf{x}}_1$（即 ${\mathbf{x}}_1\sim p_1$）。假设 $\boldsymbol{\Psi}$ 是光滑双射，则 $p_1$ 通过标准*变量变换公式*由 $p_0$ 得到：

<span id="eq-change-of-variable-density"></span>

$$
p_{1}({\mathbf{x}}_1) = p_{0}(\boldsymbol{\Psi}^{-1}({\mathbf{x}}_1)) \cdot \left| \det \left( \frac{\partial \boldsymbol{\Psi}^{-1}}{\partial {\mathbf{x}}_1} \right) \right|.
\tag{B.1.1}
$$

这里，$\tfrac{\partial \boldsymbol{\Psi}}{\partial {\mathbf{x}}}$ 是 $\boldsymbol{\Psi}$ 的 Jacobian 矩阵，记为 $\partial_{\mathbf{x}} \boldsymbol{\Psi}$。等价地，在原始坐标中，

$$
p_0({\mathbf{x}}_0) \;=\; p_1\!\big(\boldsymbol{\Psi}({\mathbf{x}}_0)\big)\;\big|\det \partial_{\mathbf{x}} \boldsymbol{\Psi}({\mathbf{x}}_0)\big|.
$$

换言之，$\boldsymbol{\Psi}$ 将密度 $p_0$ 重塑为 $p_1$。因子 $\big|\det \partial_{\mathbf{x}} \boldsymbol{\Psi}\big|$ 表示局部体积变化；由于概率质量守恒，密度需要乘以其倒数进行补偿。

作为简单情形，若 $\boldsymbol{\Psi}$ 是由可逆矩阵 ${\mathbf{A}}$ 定义的线性映射（即 ${\mathbf{x}}_1={\mathbf{A}}{\mathbf{x}}_0$），则

$$
p_1({\mathbf{x}}_1)\;=\;p_0({\mathbf{A}}^{-1}{\mathbf{x}}_1)\,\big|\det {\mathbf{A}}^{-1}\big|.
$$

概括地说，可将其理解为：

$$
\begin{array}{lccc}
   \textbf{Sample:} & {\mathbf{x}}_0 & \xrightarrow{\hspace{0.2cm} \boldsymbol{\Psi} \hspace{0.2cm}} & {\mathbf{x}}_1  \\
   \textbf{Density:} & p_{{\mathbf{x}}_0}({\mathbf{x}}_0) & \xrightarrow{\hspace{0.2cm} \boldsymbol{\Psi} \hspace{0.2cm}} & p_{{\mathbf{x}}_1}({\mathbf{x}}_1)
\end{array}
$$

**为什么[公式 (B.1.1)](01-change-of-variable.md#eq-change-of-variable-density)是变量变换公式？** 这直接来自微积分中的熟悉规则。

**单变量情形。** 设 $y = \Psi(x)$ 光滑可逆。将关于 $y$ 的积分改写为关于 $x$ 的积分，得到

$$
\int g(y)\,\mathop{}\!\mathrm{d} y = \int g(\Psi(x)) \cdot |\Psi'(x)|\,\mathop{}\!\mathrm{d} x,
$$

其中 $|\Psi'(x)|$ 补偿区间的拉伸或压缩，确保面积保持。

**多变量情形。** 对于 $\boldsymbol{\Psi}: {\mathbb{R}}^D \to {\mathbb{R}}^D$，其中 ${\mathbf{y}}=\boldsymbol{\Psi}({\mathbf{x}})$，

$$
\int g({\mathbf{y}})\,\mathop{}\!\mathrm{d}{\mathbf{y}}
= \int g(\boldsymbol{\Psi}({\mathbf{x}})) \,\big|\det(\partial_{\mathbf{x}} \boldsymbol{\Psi})\big| \,\mathop{}\!\mathrm{d}{\mathbf{x}},
$$

所以无穷小体积变换为

$$
\mathop{}\!\mathrm{d} {\mathbf{y}} = \big|\det(\partial_{\mathbf{x}} \boldsymbol{\Psi})\big|\,\mathop{}\!\mathrm{d} {\mathbf{x}}.
$$

由此得到[公式 (B.1.1)](01-change-of-variable.md#eq-change-of-variable-density)中的密度公式：

$$
\begin{aligned}
p_{{\mathbf{y}}}({\mathbf{y}})
= \int_{{\mathbb{R}}^D} \delta({\mathbf{y}} - \boldsymbol{\Psi}({\mathbf{x}}))\, p_{{\mathbf{x}}}({\mathbf{x}})\mathop{}\!\mathrm{d} {\mathbf{x}} = p_{{\mathbf{x}}}\!\big(\boldsymbol{\Psi}^{-1}({\mathbf{y}})\big)\,\Big|\det\!\left(\frac{\partial \boldsymbol{\Psi}^{-1}}{\partial {\mathbf{y}}}\right)\Big|.
\end{aligned}
$$

**复合多个双射。** 现在依次应用若干次更新。令 ${\mathbf{x}}_k = \boldsymbol{\Psi}_k({\mathbf{x}}_{k-1})$，其中 $k=1,\dots,L$；即

$$
{\mathbf{x}}_0 \xrightarrow{\ \boldsymbol{\Psi}_1\ } {\mathbf{x}}_1 \xrightarrow{\ \boldsymbol{\Psi}_2\ } \cdots
\xrightarrow{\ \boldsymbol{\Psi}_L\ } {\mathbf{x}}_L,
$$

其中每个 $\boldsymbol{\Psi}_k:{\mathbb{R}}^D \to {\mathbb{R}}^D$ 都是光滑双射。若初始状态遵循密度 $p_0$（即 ${\mathbf{x}}_0 \sim p_0$），则这一更新序列诱导密度 $p_1,\dots,p_L$，对应于 ${\mathbf{x}}_1,\dots,{\mathbf{x}}_L$。

由于每一步都保持概率质量守恒，密度按下式演化：

$$
p_{k}({\mathbf{x}}_k) \;=\; p_{k-1}({\mathbf{x}}_{k-1}) \,
   \Big|\det \partial_{{\mathbf{x}}_{k-1}} \boldsymbol{\Psi}_k({\mathbf{x}}_{k-1})\Big|^{-1},
   \quad k=1,\dots,L.
$$

递推可得，${\mathbf{x}}_L$ 处的最终密度为

<span id="eq-cov-multi-layer-nolog"></span>

$$
\begin{aligned}
p_{{\mathbf{x}}_L}({\mathbf{x}}_L)
    = p_{{\mathbf{x}}_0}({\mathbf{x}}_0)
    \cdot \prod_{k=1}^L \left| \det \left( \frac{\partial \boldsymbol{\Psi}_k}{\partial {\mathbf{x}}_{k-1}} \right) \right|^{-1}.
\end{aligned}
\tag{B.1.2}
$$

等价地，写成对数密度形式：

$$
\begin{aligned}
\log p_{{\mathbf{x}}_L}({\mathbf{x}}_L)
    = \log p_{{\mathbf{x}}_0}({\mathbf{x}}_0)
    - \sum_{k=1}^L \log \left| \det \left( \frac{\partial \boldsymbol{\Psi}_k}{\partial {\mathbf{x}}_{k-1}} \right) \right|.
\end{aligned}
$$

这一表达式反映每个变换 $\boldsymbol{\Psi}_k$ 如何拉伸或收缩体积，具体由 Jacobian 行列式刻画。沿变换路径累积这些局部体积变化，就确定了复合映射下的最终概率密度。

[公式 (B.1.2)](01-change-of-variable.md#eq-cov-multi-layer-nolog)是归一化流背后的核心原则（见[第 5.1.2 节](../../chapters/05-flow-based-perspective/01-normalizing-flows-and-neural-odes.md#subsec-node)）。

## B.1.2 连续时间极限：连续性方程 {#subsec-cov-continuity-eq}

现在从离散更新转向连续描述。假设粒子运动由时变速度场 ${\mathbf{f}}:{\mathbb{R}}^D \times [0,T]\to {\mathbb{R}}^D$ 驱动。设想让粒子 ${\mathbf{x}}_0 \sim p_0$ 经历无穷多次微小的双射更新。在每个步骤 $t$ 中，步长为 $\Delta t>0$，更新为

$$
{\mathbf{x}}_{t+\Delta t} = \boldsymbol{\Psi}({\mathbf{x}}_t)
:= {\mathbf{x}}_t + \Delta t\, {\mathbf{f}}({\mathbf{x}}_t, t).
$$

当 $\Delta t \to 0$ 时，这些更新的复合收敛到由速度场 ${\mathbf{f}}:{\mathbb{R}}^D \times [0,T]\to {\mathbb{R}}^D$ 支配的连续流：

<span id="eq-ode-f-continuity"></span>

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t} = {\mathbf{f}}({\mathbf{x}}(t), t),
    \qquad {\mathbf{x}}(0) = {\mathbf{x}}_0 \sim p_0.
\end{aligned}
\tag{B.1.3}
$$

在合适的光滑性假设下（见[附录 A](../a-differential-equations/index.md#app-de)），这一 ODE 对每个初始条件都有唯一解，从而定义确定性流映射 $\boldsymbol{\Psi}_{0\to t}:{\mathbb{R}}^D \to {\mathbb{R}}^D$。换言之，$\boldsymbol{\Psi}_{0\to t}$ 将初始状态 ${\mathbf{x}}_0$ 带到[公式 (B.1.3)](01-change-of-variable.md#eq-ode-f-continuity)在时刻 $t$ 的解：

$$
\boldsymbol{\Psi}_{0\to t}({\mathbf{x}}_0)
= {\mathbf{x}}_0 + \int_{0}^{t} {\mathbf{f}}({\mathbf{x}}(\tau), \tau)\,\mathop{}\!\mathrm{d}\tau.
$$

因此，整个分布也随之移动：初始密度 $p_0$ 被输运为新密度 $p_t$，即 ${\mathbf{x}}(t)$ 的分布。形式上，这写成*推前*：

$$
p_t = \big(\boldsymbol{\Psi}_{0\to t}\big)_{\#} p_0.
$$

当 $\boldsymbol{\Psi}_{0\to t}$ 光滑可逆时，就退化为熟悉的变量变换规则：

$$
p_t({\mathbf{x}})
= p_0\!\big(\boldsymbol{\Psi}_{t\to 0}({\mathbf{x}})\big)\,
   \big|\det \partial_{\mathbf{x}}\boldsymbol{\Psi}_{t\to 0}({\mathbf{x}})\big|
= \int \delta\!\left({\mathbf{x}} - \boldsymbol{\Psi}_{0\to t}({\mathbf{x}}_0)\right)
   p_0({\mathbf{x}}_0)\,\mathop{}\!\mathrm{d}{\mathbf{x}}_0.
$$

**连续性方程：密度如何随时间移动。** 无须为每个时刻的密度单独写一个公式，可以用关于空间 ${\mathbf{x}}$ 和时间 $t$ 的微分方程描述其连续移动。思想很简单：概率质量守恒，速度场 ${\mathbf{f}}$ 只在空间中重新分配质量。由此得到*连续性方程*：

<span id="eq-continuity-cov"></span>

$$
\begin{aligned}
\frac{\partial}{\partial t} p_t({\mathbf{x}})
    + \nabla \cdot \big( p_t({\mathbf{x}})\,{\mathbf{f}}({\mathbf{x}}, t) \big) = 0.
\end{aligned}
\tag{B.1.4}
$$

这里，散度项 $\nabla \cdot (p_t {\mathbf{f}})$ 度量流如何在局部扩张或压缩密度，确保总概率保持为 $1$。

这一偏微分方程（PDE）保证流移动粒子时概率质量守恒。实际上，它可视为变量变换公式的连续时间对应形式。

**通过变量变换公式推导连续性方程。** 从概念上看，也可以取[公式 (B.1.2)](01-change-of-variable.md#eq-cov-multi-layer-nolog)的连续时间极限得到连续性方程。不过，这里采用基于[公式 (B.1.1)](01-change-of-variable.md#eq-change-of-variable-density)的更直接推导。

**离散化与变量变换公式。** 考虑

$$
\mathbf{x}_{t+\Delta t} := \boldsymbol{\Psi}(\mathbf{x}_t) = \mathbf{x}_t + \Delta t\, {\mathbf{f}}(\mathbf{x}_t, t),
$$

这实际上是对[公式 (B.1.3)](01-change-of-variable.md#eq-ode-f-continuity)中的 ODE 在短时间区间 $\Delta t > 0$ 上作前向 Euler 离散化。映射 $\boldsymbol{\Psi}$ 关于 $\mathbf{x}_t$ 的 Jacobian 矩阵展开为

$$
\frac{\partial \boldsymbol{\Psi}}{\partial \mathbf{x}_t} = {\mathbf{I}} + \Delta t \nabla_{\mathbf{x}} {\mathbf{f}}(\mathbf{x}_t, t) + \mathcal{O}(\Delta t^2),
$$

因此其行列式满足

$$
\det\left( \frac{\partial \boldsymbol{\Psi}}{\partial \mathbf{x}_t} \right) = 1 + \Delta t\, \nabla \cdot {\mathbf{f}}(\mathbf{x}_t, t) + \mathcal{O}(\Delta t^2).
$$

这里使用标准展开 $\det({\mathbf{I}} + \Delta t\, \mathbf{A}) = 1 + \Delta t\, \operatorname{Tr}(\mathbf{A}) + \mathcal{O}(\Delta t^2)$，其中 $\Delta t \to 0$，并结合 $\nabla \cdot {\mathbf{f}} = \operatorname{Tr}(\nabla_{\mathbf{x}} {\mathbf{f}})$。

应用变量变换公式，对数密度按下式演化：

$$
\begin{aligned}
\log p_{t+\Delta t}(\mathbf{x}_{t+\Delta t}) = \log p_t(\mathbf{x}_t) - \Delta t\, \nabla \cdot {\mathbf{f}}(\mathbf{x}_t, t) + \mathcal{O}(\Delta t^2).
\end{aligned}
$$

即

<span id="eq-log-density-diff"></span>

$$
\begin{aligned}
\log p_{t+\Delta t}(\mathbf{x}_{t+\Delta t}) - \log p_t(\mathbf{x}_t) = - \Delta t\, \nabla \cdot {\mathbf{f}}(\mathbf{x}_t, t) + \mathcal{O}(\Delta t^2).
\end{aligned}
\tag{B.1.5}
$$

**使用 Taylor 展开。** 现在对左侧作多变量 Taylor 展开：

$$
\begin{aligned}
&\log p_{t+\Delta t}({\mathbf{x}}_{t+\Delta t})
    - \log p_t({\mathbf{x}}_t)
    \\= &\Delta t\, \partial_t \log p_t({\mathbf{x}}_t)
    + ({\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t)^\top \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t)
    + \mathcal{O}(\Delta t^2).
\end{aligned}
$$

代入 ${\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t = {\mathbf{f}}({\mathbf{x}}_t, t)\Delta t$，得到：

$$
\begin{aligned}
&\log p_{t+\Delta t}({\mathbf{x}}_{t+\Delta t})
    - \log p_t({\mathbf{x}}_t)
    \\= & \Delta t\, \partial_t \log p_t({\mathbf{x}}_t)
    + \Delta t\, {\mathbf{f}}({\mathbf{x}}_t, t)^\top \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t)
    + \mathcal{O}(\Delta t^2).
\end{aligned}
$$

与[公式 (B.1.5)](01-change-of-variable.md#eq-log-density-diff)逐项匹配，并令 $\Delta t \to 0$，可得

$$
\partial_t \log p_t({\mathbf{x}}_t)
    = - \nabla_{{\mathbf{x}}_t} \cdot {\mathbf{f}}({\mathbf{x}}_t, t)
    - {\mathbf{f}}({\mathbf{x}}_t, t)^\top \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t).
$$

取指数并使用乘积法则，得到连续性方程。

**确定性输运的 Lagrangian 与 Eulerian 视角。** 同一个确定性输运有两种互补描述。*Lagrangian* 视角通过流映射 $\boldsymbol{\Psi}_{0\to t}$ 跟踪单个粒子，*Eulerian* 视角则描述固定空间位置处的速度 ${\mathbf{f}}({\mathbf{x}},t)$ 和密度 $p_t({\mathbf{x}})$。在合适的正则性条件下，这些描述一致：给定速度场生成粒子流，其推前密度满足连续性方程。

不过，其中存在一个重要的不对称性。一旦速度场和初始密度固定，所得流与密度演化就确定了；但仅给定密度路径，通常不能确定唯一速度场或粒子流。

**从速度到流与密度。** 到目前为止，我们假设速度场 ${\mathbf{f}}$ 已给定。粒子 ODE

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t} = {\mathbf{f}}({\mathbf{x}}(t), t)
$$

描述每个粒子如何移动，而密度 PDE

$$
\partial_t p_t + \nabla \!\cdot \!\big(p_t {\mathbf{f}}_t\big) = 0
$$

描述整个粒子分布如何演化。这两种视角相互联系：按 ODE 移动粒子，会自动产生满足 PDE 的密度。此时粒子流 $\boldsymbol{\Psi}_{0\to t}$ 唯一确定：从 ${\mathbf{x}}(0)\sim p_0$ 出发，每条轨迹 ${\mathbf{x}}(t)$ 都固定，所得密度 $p_t$ 遵循连续性方程。这里，粒子动力学与密度动力学完全一致。

**从密度路径到速度：非唯一性。** 如果只从密度路径 $t \mapsto p_t$ 出发（例如流匹配中的[第 5.3.2 节](../../chapters/05-flow-based-perspective/03-probability-paths-and-velocities.md#subsec-fm-instantiations)），速度场就不再唯一确定。例如，若向量场 $\mathbf w_t$ 满足

$$
\nabla_{{\mathbf{x}}}\cdot\!\big(p_t({\mathbf{x}})\,\mathbf w_t({\mathbf{x}})\big)=0
\qquad\text{(no net flux w.r.t. $p_t$),}
$$

那么 ${\mathbf{f}}_t$ 和 ${\mathbf{f}}_t+\mathbf w_t$ 都会产生相同的密度演化。因此，一条密度路径可能对应许多不同的流，选择特定粒子流 $\boldsymbol{\Psi}_{0\to t}$，就相当于在这些可能性中选定一个速度场。

并非每条给定路径 $p_t$ 都能由粒子在某个速度场下的运动产生。连续性方程（[公式 (B.1.4)](01-change-of-variable.md#eq-continuity-cov)）为密度路径能否“由流生成”提供一致性检验。称 $p_t$ 是*可实现的*（或*由 ${\mathbf{f}}$ 生成*），是指存在速度场 ${\mathbf{f}}$，使遵循下式的粒子

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t} = {\mathbf{f}}({\mathbf{x}}(t), t)
$$

恰好产生密度 $p_t$，其所用的流映射为 $\boldsymbol{\Psi}_{0\to t}$。也就是说，当 $p_t$ 与 ${\mathbf{f}}$ 共同满足[公式 (B.1.4)](01-change-of-variable.md#eq-continuity-cov)时，可实现性成立。

直观上，可实现性意味着 $p_t$ 随时间的快照能够由粒子在某个速度场下的运动解释，而非任意的分布序列。

当这一条件成立时，密度 $p_t$ 就是初始密度 $p_0$ 沿流映射 $\boldsymbol{\Psi}_{0\to t}$ 的推前。此时，熟悉的变量变换公式适用：

$$
\begin{aligned}
p_t = \big(\boldsymbol{\Psi}_{0\to t}\big)_{\#} p_0
= p_0\!\big(\boldsymbol{\Psi}_{t\to 0}({\mathbf{x}})\big)\,
   \big|\det \partial_{\mathbf{x}}\boldsymbol{\Psi}_{t\to 0}({\mathbf{x}})\big|
= \int \delta \left({\mathbf{x}} - \boldsymbol{\Psi}_{0\to t}({\mathbf{x}}_0)\right)
   p_0({\mathbf{x}}_0)\,\mathop{}\!\mathrm{d}{\mathbf{x}}_0.
\end{aligned}
$$

**（选读）条件化。** 若引入额外条件变量 ${\mathbf{z}} \sim \pi({\mathbf{z}})$，相同推理适用于每个固定的 ${\mathbf{z}}$：

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t}={\mathbf{v}}_t({\mathbf{x}}(t)|{\mathbf{z}})
$$

其推前为 $p_t(\cdot|{\mathbf{z}})=({\boldsymbol{\Psi}}_{0\to t}(\cdot;{\mathbf{z}}))_{\#}p_0$，连续性方程为

$$
\partial_t p_t({\mathbf{x}}|{\mathbf{z}})+\nabla\!\cdot\!\big(p_t({\mathbf{x}}|{\mathbf{z}})\,{\mathbf{v}}_t({\mathbf{x}}|{\mathbf{z}})\big)=0
$$

于是边缘密度为

$$
p_t({\mathbf{x}})=\int p_t({\mathbf{x}}|{\mathbf{z}})\pi({\mathbf{z}})\mathop{}\!\mathrm{d}{\mathbf{z}}.
$$

## B.1.3 随机过程：Fokker–Planck 方程 {#subsec-cov-fpe}

加入噪声后，动力学遵循[公式 (A.2.3)](../a-differential-equations/02-stochastic-differential-equations.md#eq-sde)中的 SDE：

$$
\mathop{}\!\mathrm{d} {\mathbf{x}}(t) = {\mathbf{f}}({\mathbf{x}}(t), t)\mathop{}\!\mathrm{d} t + g(t)\mathop{}\!\mathrm{d} {\mathbf{w}}(t).
$$

此时，密度 $p_t({\mathbf{x}})$ 满足 Fokker–Planck 方程：

$$
\begin{aligned}
\frac{\partial p_t({\mathbf{x}})}{\partial t}
&= - \nabla \cdot \left( {\mathbf{f}}({\mathbf{x}}, t)\, p_t({\mathbf{x}}) \right)
+ \frac{1}{2} g^2(t)\, \Delta p_t({\mathbf{x}})
\\& = - \nabla \cdot \left(  \left({{\mathbf{f}}({\mathbf{x}}, t)}-{\frac{1}{2} g^2(t) \nabla_{\mathbf{x}} \log p_t({\mathbf{x}})}\right) p_t({\mathbf{x}})\right).
\end{aligned}
$$

$\Delta p_t = \nabla \cdot \nabla_{\mathbf{x}} p_t$ 是 Laplace 算子。在第一个等式中，第一项描述由确定性漂移 ${\mathbf{f}}$ 产生的概率质量输运，第二项描述随机噪声导致的密度扩散，其方差与 $\tfrac{1}{2} g^2(t)$ 成正比。第二个等式将相同的密度演化改写为连续性方程，其速度为

$$
{\mathbf{f}}({\mathbf{x}},t)
-
\frac{1}{2}g^2(t)\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}}).
$$

因此，由这一速度驱动的确定性 ODE，与该 SDE 具有相同的各单时刻边缘密度 $p_t$。这就是本书使用的概率流 ODE。

**注。**

关键在于，SDE 与 PF-ODE 的*密度路径*相同，但不同时刻样本之间的耦合通常不同。在任意固定时刻 $t$，二者产生相同分布 $p_t$；等价地，其粒子“快照”具有相同直方图。

但跨时间观察时，其样本路径不同。对于 PF-ODE，一旦状态 ${\mathbf{x}}_s$ 固定，后续状态便由 ${\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)$ 确定。对于 SDE，时刻 $s$ 后会注入新的 Brownian 噪声，因此即使以 ${\mathbf{x}}_s$ 为条件，后续状态仍然随机。

因此，即使每个单独时刻的边缘密度相同，两种过程也可能具有不同的转移分布、时间相关性和路径事件。PF-ODE 复现的是 SDE 的*边缘密度演化*，而非其完整的随机路径分布。示意见[图 4.5](../../chapters/04-score-sde-framework/02-matching-marginals.md#fig-score-sde-three-dynamics)。

Fokker–Planck 方程的推导较为复杂，参见[第 C.1.4 节](../c-ito-calculus/01-ito-formula.md#subsec-rigorous-proof-fpe)。
