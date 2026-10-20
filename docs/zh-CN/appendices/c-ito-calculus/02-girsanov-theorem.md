# C.2 测度的变量替换：扩散模型中的 Girsanov 定理 {#sec-girsanov-theorem}

扩散模型通过 SDE 将简单噪声转变为丰富的数据分布。这个变换背后有一个深刻的思想：只修改 SDE 的确定性部分（漂移），同时保留其底层随机性，就可以重新解释随机性。Girsanov 定理正是在这里发挥作用。

**基本思想。** 考虑一条观测到的连续轨迹，它描述数据从时间 $t=0$ 到 $t=T$ 的演化，记为 ${\mathbf{x}}_{0:T}:=\{ {\mathbf{x}}_t|t \in [0,T] \}$。Girsanov 定理回答如下基本问题：

<span id="question-c-2-1"></span>

**问题 C.2.1**

对于这条观测路径，若假设它由某个 SDE 生成，与假设它由另一个 SDE 生成相比，其似然有何不同？

我们比较两种假设的生成模型，它们用于生成同一条轨迹，具有相同的初始分布和扩散系数，但确定性的“推动作用”，即漂移函数不同。假设 ${\mathbf{x}}_0$ 在这两个假定的生成过程中具有相同的初始分布。

为建立直观认识，可以将 ${\mathbf{x}}_{0:T}$ 想象成画在纸上的一条曲折线。一种假设根据漂移 ${\mathbf{f}}$ 为轨迹分配概率，另一种假设采用不同的漂移 $\tilde{{\mathbf{f}}}$，但扩散系数相同。Girsanov 定理比较相应的路径测度，并量化改变漂移会如何改变分配给同一批轨迹的相对权重。

**设定。** 令 $P_{{\mathbf{f}}}$ 和 $P_{\tilde{{\mathbf{f}}}}$ 表示两个 SDE 诱导的路径测度，它们具有相同的初始分布和扩散系数，但漂移不同：

$$
\begin{aligned}
\mathop{}\!\mathrm{d}{\mathbf{x}}_t
    &= {\mathbf{f}}({\mathbf{x}}_t,t)\mathop{}\!\mathrm{d} t + g(t)\mathop{}\!\mathrm{d}{\mathbf{w}}_t,
    \\
    \mathop{}\!\mathrm{d}{\mathbf{x}}_t
    &= \tilde{{\mathbf{f}}}({\mathbf{x}}_t,t)\mathop{}\!\mathrm{d} t + g(t)\mathop{}\!\mathrm{d}{\mathbf{w}}_t.
\end{aligned}
$$

为简单起见，假设在所讨论的区间上 $g(t)>0$，并满足 Girsanov 定理所需的标准可积性条件。取 $P_{\tilde{{\mathbf{f}}}}$ 为参考路径测度，令 ${\mathbf{w}}_t$ 表示该参考测度下的 Brownian 运动。定义

$$
\boldsymbol{\delta}_t
:=
{\mathbf{f}}({\mathbf{x}}_t,t)-\tilde{{\mathbf{f}}}({\mathbf{x}}_t,t).
$$

**Girsanov 似然比。** Girsanov 定理给出两个路径测度之间的 Radon–Nikodym 导数：

$$
\frac{\mathop{}\!\mathrm{d} P_{{\mathbf{f}}}}{\mathop{}\!\mathrm{d} P_{\tilde{{\mathbf{f}}}}}({\mathbf{x}}_{0:T})
=
\exp\left(
\int_0^T
\boldsymbol{\delta}_t^\top g(t)^{-1}\mathop{}\!\mathrm{d}{\mathbf{w}}_t
-
\frac{1}{2}
\int_0^T
\big\|g(t)^{-1}\boldsymbol{\delta}_t\big\|^2
\mathop{}\!\mathrm{d} t
\right).
$$

这一紧凑公式是包含两个积分的指数表达式，其中第一个是 Itô 积分，第二个是普通的 Riemann 积分。这个比值在扩散模型中非常重要，使我们能够比较不同的随机动力学，并将路径空间中的差异与基于似然的训练目标联系起来。

可以将 Girsanov 定理理解为*测度*的变量替换公式。正如微积分中的变量替换通过 Jacobian 行列式在不同坐标系之间变换积分，当漂移发生改变而扩散保持相同时，Girsanov 定理给出了相应的因子（Radon–Nikodym 导数），用来在两个随机过程之间变换概率或期望。

## C.2.1 Girsanov 定理：连接似然训练与 score 匹配 {#subsec-girsanov}

在理解 Girsanov 定理如何关联同一路径在不同漂移假设下的似然后，我们进一步讨论它对扩散模型的意义。

回顾扩散模型中的前向 SDE：

$$
\mathop{}\!\mathrm{d} {\mathbf{x}}_t = {\mathbf{f}}({\mathbf{x}}_t, t) \mathop{}\!\mathrm{d} t + g(t) \mathop{}\!\mathrm{d} {\mathbf{w}}_t,
$$

它在完整轨迹上诱导出*路径分布* $P$，对应轨迹 ${\mathbf{x}}_{0:T} := \{{\mathbf{x}}_t\}_{t=0}^T$（即该过程在整个时间区间上的联合分布）。逆时间 SDE 由可学习的 score 函数 ${\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t)$ 参数化，形式为

$$
\mathop{}\!\mathrm{d} {\mathbf{x}}_t = \big[{\mathbf{f}}({\mathbf{x}}_t, t) - g^2(t) {\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t)\big] \mathop{}\!\mathrm{d} t
+ g(t) \mathop{}\!\mathrm{d} \bar{{\mathbf{w}}}_t,
$$

它又定义了轨迹上的另一个路径分布 $P_{\boldsymbol{\phi}}$。

**扩散模型中的两个概念。** 两个随机过程可以在两个不同层面上一致，有必要将它们区分开：

-   **概念 1：边缘分布匹配。** 两个过程具有相同的分布 $p_t({\mathbf{x}}_t)$，对每个固定时间 $t$ 都如此。因此，它们在各个时刻的“快照”相同。

-   **概念 2：联合路径分布匹配。** 两个过程在整条轨迹上具有相同的概率分布。这是更强的概念，还匹配了不同时刻状态之间的耦合方式，包括转移规律和时间依赖。

匹配完整路径分布会自动保证边缘匹配，因为每个 $p_t$ 都可以通过将路径分布投影到时刻 $t$ 的状态而得到。反过来通常不成立：两个过程可以在每个时刻具有相同的边缘分布，却在这些边缘之间产生不同的耦合。SDE 及其对应的 PF-ODE 就是一个重要例子：它们具有相同的边缘密度路径，但样本轨迹和路径分布通常不同。

**Girsanov 定理将 score 误差与路径分布不匹配联系起来。** 精确的逆时间 SDE 满足上述更强的概念。在标准正则条件下，若从精确的终端边缘分布 $p_T$ 初始化，并使用真实 score $\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}})$，它就是前向扩散的时间反演，因此能够重现完整的时间反演路径分布。

令 $P^{\mathrm{rev}}$ 表示这一精确逆时间路径分布，$P_{\boldsymbol{\phi}}$ 表示学得的逆时间路径分布，后者以 ${\mathbf{s}}_{\boldsymbol{\phi}}$ 替代真实 score，并从 $p_{\mathrm{prior}}$ 初始化。将路径空间 KL散度分解为初始分布的不匹配和条件路径分布的不匹配，再把两个过程都表示在同一个递增的逆时间时钟下，对后一项应用 Girsanov 定理，可在原始扩散时间标记 $t$ 下得到

<span id="eq-girsanov-kl"></span>

$$
\begin{aligned}
\mathcal{D}_{\mathrm{KL}}
\bigl(P^{\mathrm{rev}}\|P_{\boldsymbol{\phi}}\bigr)
=
&\;
\mathcal{D}_{\mathrm{KL}}
\bigl(p_T\|p_{\mathrm{prior}}\bigr)
\\
&+
\frac{1}{2}
\int_0^T
g^2(t)
{\mathbb{E}}_{{\mathbf{x}}_t\sim p_t}
\left[
\left\|
{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t)
-
\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}}_t)
\right\|^2
\right]
\mathop{}\!\mathrm{d} t .
\end{aligned}
\tag{C.2.1}
$$

第一项衡量真实终端分布与生成初始化所用先验之间的不匹配，当 $p_T=p_{\mathrm{prior}}$ 时消失。第二项衡量学得的逆向漂移与精确逆向漂移之间累积的不匹配。

因此，采用特定似然权重 $g^2(t)$ 的 score 匹配具有直接的路径空间解释：除固定的终端分布项外，最小化加权 score 误差就是最小化精确逆向路径分布与学得的逆向路径分布之间的 KL散度。如果用去噪 score 匹配替代边缘 score 匹配，相应目标只相差一个与参数无关的常数。其他时间权重通常不对应同一个路径空间 KL散度。

**与数据似然的联系。** 路径空间 KL散度还控制着生成边缘分布与数据边缘分布之间的差异。由于取轨迹的端点是一种可测投影，且 KL散度在边缘化后不会增大，

<span id="eq-girsanov-likelihood"></span>

$$
\begin{aligned}
\mathcal{D}_{\mathrm{KL}}
\bigl(p_{\mathrm{data}}\|p_{\boldsymbol{\phi}}\bigr)
\leq
\mathcal{D}_{\mathrm{KL}}
\bigl(P^{\mathrm{rev}}\|P_{\boldsymbol{\phi}}\bigr),
\end{aligned}
\tag{C.2.2}
$$

其中 $p_{\boldsymbol{\phi}}$ 表示数据时刻的生成边缘分布。

将其与[式（C.2.1）](02-girsanov-theorem.md#eq-girsanov-kl)结合可知，似然加权的 score 匹配加上终端分布不匹配，给出了数据侧 KL散度的上界；因此，除一个与 $\boldsymbol{\phi}$ 无关的常数外，它也给出了期望负对数似然的上界（[Song et al., 2021](../../references.md#ref-song2021maximum)）。
