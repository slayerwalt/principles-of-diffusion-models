# 9.2 DDIM {#sec-revisit-ddim}

*Denoising Diffusion Implicit Models*（DDIM）（[Song et al., 2020](../../references.md#ref-song2020denoising)）是最早且应用最广泛的扩散采样加速方法之一。DDIM 最初从概率视角提出，但其确定性更新也能通过 PF-ODE 得到特别清晰的解释。因此，本章先在统一的数值 ODE 求解框架下介绍 DDIM，将其最初的变分视角留到[第 9.2.3 节](02-ddim.md#subsec-ddim)。

[第 9.1 节](01-prologue.md#sec-prologue-solvers)的引言留下了一个具体的数值问题：在解析保留已知调度相关因子后，应如何近似剩余积分内的神经网络预测？DDIM 选择最简单的答案：*使用该步起点处的值*。下面先推导相应更新，再澄清它与经典 Euler 积分，以及后文高阶扩散求解器的关系。

## 9.2.1 DDIM 作为一阶扩散求解器 {#subsec-ddim-different-para}

考虑一个采样步，从起点[^1] $\tilde{\mathbf{x}}_s$（时刻 $s$）推进到 $t<s$。在噪声预测下，[式 (9.1.7)](01-prologue.md#eq-variation-of-constants)给出精确经验 PF-ODE 流

$$
\begin{aligned}
\widetilde{\boldsymbol{\Psi}}_{s\to t}(\tilde{\mathbf{x}}_s)
=
\mathcal E(s\shortrightarrow t)\tilde{\mathbf{x}}_s
+
\frac{1}{2}
\int_s^t
\frac{g^2(\tau)}{\sigma_\tau}
\mathcal E(\tau\shortrightarrow t)
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)
\mathop{}\!\mathrm{d}\tau .
\end{aligned}
$$

调度相关因子解析已知；积分中唯一未知的量，是沿未知轨迹计算的神经网络预测 ${\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)$。

从 PF-ODE 视角看，确定性 DDIM 使用最简单的一阶近似：将噪声预测固定为该步起点的值，

$$
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)
\approx
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s),\quad\text{for}\quad \tau\in[t,s],
$$

同时在积分内保留已知的调度相关因子。由此得到

$$
\begin{aligned}
\tilde{\mathbf{x}}_t
=
\mathcal E(s\shortrightarrow t)\tilde{\mathbf{x}}_s
+
\left[
\frac{1}{2}
\int_s^t
\frac{g^2(\tau)}{\sigma_\tau}
\mathcal E(\tau\shortrightarrow t)
\mathop{}\!\mathrm{d}\tau
\right]
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s).
\end{aligned}
$$

利用 $f(t)=\alpha_t'/\alpha_t$，积分因子简化为 $\mathcal E(s\shortrightarrow t)=\alpha_t/\alpha_s$。剩余调度相关系数也具有方便的闭式形式：$\tfrac{1}{2}\tfrac{g^2(\tau)}{\sigma_\tau} \mathcal E(\tau\shortrightarrow t) = \alpha_t \tfrac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d}\tau} \bigl(\tfrac{\sigma_\tau}{\alpha_\tau}\bigr)$。因此，一旦在该步内固定神经网络预测，就能根据噪声调度精确计算剩余积分。

<span id="ddim-euler"></span>

**命题 9.2.1：DDIM 作为一阶因子分解 EI 更新**

对于 $s>t$，确定性 DDIM 给出

<span id="eq-ddim-euler"></span>

$$
\begin{aligned}
\tilde{\mathbf{x}}_t
=
\frac{\alpha_t}{\alpha_s}\tilde{\mathbf{x}}_s
+
\alpha_t
\left(
\frac{\sigma_t}{\alpha_t}
-
\frac{\sigma_s}{\alpha_s}
\right)
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s).
\end{aligned}
\tag{9.2.1}
$$

**命题证明。**

将 $\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d}\tau}(\sigma_\tau/\alpha_\tau)$ 从 $s$ 积分到 $t$ 即可得到结论。

因此，DDIM 所作的近似非常具体：在整个区间内保留已知调度的演化，只忽略神经网络预测的变化。

干净数据预测下也有类似推导。精确经验 PF-ODE 流可写为

$$
\begin{aligned}
\widetilde{\boldsymbol{\Psi}}_{s\to t}(\tilde{\mathbf{x}}_s)
=
\frac{\sigma_t}{\sigma_s}\tilde{\mathbf{x}}_s
+
\sigma_t
\int_s^t
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d}\tau}
\left(
\frac{\alpha_\tau}{\sigma_\tau}
\right)
{\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)
\mathop{}\!\mathrm{d}\tau.
\end{aligned}
$$

使用与上面相同的一阶近似，将 ${\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)$ 固定为时刻 $s$ 的值。随后精确计算剩余的调度相关积分，得到

$$
\tilde{\mathbf{x}}_t
=
\frac{\sigma_t}{\sigma_s}\tilde{\mathbf{x}}_s
+
\alpha_s
\left(
\frac{\alpha_t}{\alpha_s}
-
\frac{\sigma_t}{\sigma_s}
\right)
{\mathbf{x}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s).
$$

因此，无论在噪声预测 PF-ODE 中冻结噪声预测，还是在干净数据 PF-ODE 中冻结干净数据预测，都得到相同的 DDIM 更新。这一对应关系将在后面的 DPM-Solver 和 DPM-Solver`++` 中再次出现，二者分别从噪声预测与干净数据预测表述构造高阶近似。

**等价形式与解释。** 在 oracle 层面，标准预测目标通过[式 (6.3.1)](../06-unified-perspective/03-equivalent-parameterizations.md#eq-predictions-equivalence)逐点代数等价。因此，上面从噪声预测推导的确定性 DDIM 更新，可以在起点 $s$ 用干净数据预测或 score 预测改写，而不改变有限步更新。以下推论汇总这些等价表达式。

**推论 9.2.1：确定性 DDIM 更新的等价形式**

令 $s>t$，考虑状态 $\tilde{\mathbf{x}}_s$，其时刻为 $s$。oracle 确定性 DDIM 更新具有以下等价表达式

<span id="eq-ddim-update-all"></span>

$$
\begin{aligned}
\tilde{\mathbf{x}}_t
&=
\frac{\alpha_t}{\alpha_s}\tilde{\mathbf{x}}_s
+
\alpha_t
\left(
\frac{\sigma_t}{\alpha_t}
-
\frac{\sigma_s}{\alpha_s}
\right)
{\boldsymbol{\epsilon}}^*(\tilde{\mathbf{x}}_s,s)
\\
&=
\frac{\sigma_t}{\sigma_s}\tilde{\mathbf{x}}_s
+
\alpha_s
\left(
\frac{\alpha_t}{\alpha_s}
-
\frac{\sigma_t}{\sigma_s}
\right)
{\mathbf{x}}^*(\tilde{\mathbf{x}}_s,s)
\\
&=
\frac{\alpha_t}{\alpha_s}\tilde{\mathbf{x}}_s
+
\sigma_s^2
\left(
\frac{\alpha_t}{\alpha_s}
-
\frac{\sigma_t}{\sigma_s}
\right)
{\mathbf{s}}^*(\tilde{\mathbf{x}}_s,s)
\\
&=
\alpha_t
\underset{\substack{
\approx{\mathbf{x}}_{\boldsymbol{\phi}^\times}\\
\text{estimated clean}
}}{
\underbrace{{\mathbf{x}}^*(\tilde{\mathbf{x}}_s,s)}
}
+
\sigma_t
\underset{\substack{
\approx{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}\\
\text{estimated noise}
}}{
\underbrace{{\boldsymbol{\epsilon}}^*(\tilde{\mathbf{x}}_s,s)}
}.
\end{aligned}
\tag{9.2.2}
$$

[式 (9.2.2)](02-ddim.md#eq-ddim-update-all)的最后一个恒等式提供了特别直观的 DDIM 视角。从 $\tilde{\mathbf{x}}_s$ 出发，DDIM 固定在时刻 $s$ 推断出的干净数据与噪声估计，再使用目标调度系数重新组合：

$$
\tilde{\mathbf{x}}_t
=
\alpha_t{\mathbf{x}}^*(\tilde{\mathbf{x}}_s,s)
+
\sigma_t{\boldsymbol{\epsilon}}^*(\tilde{\mathbf{x}}_s,s).
$$

这与前向 Gaussian 路径 ${\mathbf{x}}_\tau=\alpha_\tau{\mathbf{x}}_0+\sigma_\tau{\boldsymbol{\epsilon}}$ 的共享噪声耦合相呼应：给定固定样本对 $({\mathbf{x}}_0,{\boldsymbol{\epsilon}})$，从一个时刻移动到另一个时刻，只改变系数 $(\alpha_\tau,\sigma_\tau)$。DDIM 使用该步起点处可获得的干净数据与噪声估计，应用同一思想。

在条件层面，${\mathbf{x}}_0$ 和 ${\boldsymbol{\epsilon}}$ 固定时，这一重组精确遵循相应共享噪声路径。但在边缘层面，${\mathbf{x}}^*(\tilde{\mathbf{x}}_s,s)$ 和 ${\boldsymbol{\epsilon}}^*(\tilde{\mathbf{x}}_s,s)$ 是根据 $\tilde{\mathbf{x}}_s$ 推断的后验估计。因此，有限 DDIM 步一般只是边缘 PF-ODE 流的近似，并非从 $p_t$ 精确采样。

<span id="fig-euler"></span>

![有限步 DDIM 与连续经验概率流 ODE 的比较](../../../en/assets/chapter-09/ddim-pf-ode-flow.png)

**图 9.1：** **有限步 DDIM 与经验 PF-ODE 流。** 从 $\tilde{\mathbf{x}}_s$ 出发，经验 PF-ODE 连续演化到 $\widetilde{\boldsymbol{\Psi}}_{s\to t}(\tilde{\mathbf{x}}_s)$，而 DDIM 在整个区间内保持起始时刻的神经网络预测，同时解析演化已知调度。二者差异就是[式 (9.2.3)](02-ddim.md#eq-ddim-local-defect)中的单步离散化误差。

来源：作者绘制。

**选择固定哪一种预测为何重要？** 在任意固定状态与时刻，标准预测参数化通过[式 (6.3.1)](../06-unified-perspective/03-equivalent-parameterizations.md#eq-predictions-equivalence)代数等价。但有限步求解器通过在区间内固定某一种预测，引入额外近似。只有转换后的预测沿所得近似轨迹也保持常数时，这种冻结近似才会在参数化变化后保持不变。

DDIM 提供了一个特别清晰的例子。令

$$
\widehat{\mathbf{x}}_0
:=
{\mathbf{x}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s),
\qquad
\widehat{\boldsymbol{\epsilon}}
:=
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s),
$$

其中，如果模型只直接预测一个量，另一个量就通过[式 (6.3.1)](../06-unified-perspective/03-equivalent-parameterizations.md#eq-predictions-equivalence)中的代数转换定义。因此，由构造可得 $\tilde{\mathbf{x}}_s=\alpha_s\widehat{\mathbf{x}}_0+\sigma_s\widehat{\boldsymbol{\epsilon}}$。

固定这一初始时刻的干净数据／噪声对，就定义了整个采样步上的 DDIM 插值：

$$
\tilde{\mathbf{x}}_\tau^{\mathrm{DDIM}}
=
\alpha_\tau\widehat{\mathbf{x}}_0
+
\sigma_\tau\widehat{\boldsymbol{\epsilon}},
\qquad
\tau\in[t,s].
$$

这不需要额外模型计算：一旦 $(\widehat{\mathbf{x}}_0,\widehat{\boldsymbol{\epsilon}})$ 在时刻 $s$ 固定，调度系数 $(\alpha_\tau,\sigma_\tau)$ 在每个中间时刻 $\tau$ 都已知。在 $\tau=s$ 处，该表达式恢复起始状态；在 $\tau=t$ 处，给出 DDIM 更新。中间值仅描述 DDIM 近似诱导的有限步轨迹。

噪声预测与干净数据预测的特殊作用现在一目了然。沿这条轨迹，

$$
\frac{
\tilde{\mathbf{x}}_\tau^{\mathrm{DDIM}}
-
\sigma_\tau\widehat{\boldsymbol{\epsilon}}
}{
\alpha_\tau
}
=
\widehat{\mathbf{x}}_0,
\qquad
\frac{
\tilde{\mathbf{x}}_\tau^{\mathrm{DDIM}}
-
\alpha_\tau\widehat{\mathbf{x}}_0
}{
\sigma_\tau
}
=
\widehat{\boldsymbol{\epsilon}}.
$$

因此，代数转换后的干净数据与噪声预测沿 DDIM 轨迹同时保持常数。所以，固定 ${\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau) \approx{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s)$ 或固定 ${\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau) \approx{\mathbf{x}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s)$ 会得到相同的 DDIM 更新。在这一意义上，$\widehat{\mathbf{x}}_0$ 与 $\widehat{\boldsymbol{\epsilon}}$ 是 DDIM 共享噪声路径的两个固定分量，$(\alpha_\tau,\sigma_\tau)$ 则提供已知的时变权重。

不过，score 与速度的表现不同。DDIM 固定推断出的干净数据／噪声对 $(\widehat{\mathbf{x}}_0,\widehat{\boldsymbol{\epsilon}})$，但由该样本对得到的 score 和速度一般随时间变化。

例如，将固定（在时刻 $s$）的噪声估计转换为中间时刻 $\tau$ 的 score 预测，得到

$$
{\mathbf{s}}(\tilde{\mathbf{x}}_\tau^{\mathrm{DDIM}},\tau)
=
-\frac{\widehat{\boldsymbol{\epsilon}}}{\sigma_\tau}
=
\frac{\sigma_s}{\sigma_\tau}
{\mathbf{s}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s).
$$

因此，即使 $\widehat{\boldsymbol{\epsilon}}$ 固定，对应 score 也通常随 $\tau$ 变化，这是通过因子 $1/\sigma_\tau$ 实现的。直接施加

$$
{\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)
\approx
{\mathbf{s}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s),
\qquad
\tau\in[t,s],
$$

就作出了不同的有限步近似，通常产生不同的一阶因子分解求解器。类似地，固定的干净数据／噪声对诱导速度

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d}\tau}
\tilde{\mathbf{x}}_\tau^{\mathrm{DDIM}}
=
\alpha_\tau'\widehat{\mathbf{x}}_0
+
\sigma_\tau'\widehat{\boldsymbol{\epsilon}}.
$$

这一速度一般随 $\tau$ 变化，因为调度导数 $\alpha_\tau'$ 和 $\sigma_\tau'$ 会变化。如果直接固定原始速度预测，即 ${\mathbf{v}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau) \approx {\mathbf{v}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s)$，就得到原始扩散时间下的普通 Euler；它与 DDIM 的关系将在[第 9.2.2 节](02-ddim.md#subsec-ddim-classical-euler)讨论。

**说明。**

预测参数化逐点代数等价，但有限步冻结不一定能与随时间变化的转换交换顺序。DDIM 固定一对相互一致的起始时刻干净数据／噪声估计；对应 score 和完整速度一般在该步内变化。因此，直接冻结原始 score 或速度，会定义不同的一阶方法。[第 9.2.2 节](02-ddim.md#subsec-ddim-classical-euler)将定量分析速度情形。

## 9.2.2 与 Euler 型方法的关系 {#subsec-ddim-classical-euler}

上述讨论说明 DDIM 是一阶方法，但它与经典 Euler 方法的关系需要仔细区分。一旦将已知扩散调度与学习到的预测分开，便会根据*在有限步内固定什么量*而产生不同的一阶格式。

我们比较三种只计算一次模型的构造：

1.  *普通 Euler*，固定完整 PF-ODE 速度；

2.  *指数 Euler*，解析处理已知线性传播，但固定剩余的系数加权残差；

3.  *DDIM*，固定起始时刻的干净数据／噪声估计，同时保留它们在已知扩散调度下的有限步演化。

从同一 DDIM 更新的两个等价视角，最容易看清这些区别。干净数据／噪声形式清楚展示它与普通 Euler 的差别，因子分解半线性形式则澄清它与指数 Euler 的关系。随后确定剩余离散化误差，并精确说明：在作出与调度相适应的坐标变换后，DDIM 本身可在何种意义下视为 Euler 方法。

**DDIM 与普通 Euler。** DDIM 和普通 Euler 使用相同的起始时刻模型信息，但在有限步内传播这些信息的方式不同。普通 Euler 局部近似完整 PF-ODE 速度，而 DDIM 在固定模型预测的同时，保留已知扩散调度的有限步演化。因此，对于光滑调度，两种更新在一阶上一致，一般相差 $\mathcal O((\Delta s)^2)$ 阶项。下面明确写出这一差异。

考虑从时刻 $s$ 到 $t<s$ 的一个后向步骤，取 $\Delta s:=s-t>0$，并假设起始时刻的干净数据与噪声预测在代数上一致：

$$
\tilde{\mathbf{x}}_s
=
\alpha_s
{\mathbf{x}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s)
+
\sigma_s
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s).
$$

在时刻 $s$，相应经验 PF-ODE 速度为

$$
{\mathbf{v}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s)
=
\alpha_s'
{\mathbf{x}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s)
+
\sigma_s'
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s).
$$

普通 Euler 在该步内固定这一完整速度，即 $\tau\in[t,s]$，得到：

$$
\tilde{\mathbf{x}}_t^{\mathrm{Euler}}
=
\tilde{\mathbf{x}}_s
-
\Delta s\,
{\mathbf{v}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s).
$$

DDIM 则固定起始时刻的干净数据／噪声预测对，同时将已知调度系数从 $(\alpha_s,\sigma_s)$ 演化到 $(\alpha_t,\sigma_t)$：

$$
\tilde{\mathbf{x}}_t^{\mathrm{DDIM}}
=
\tilde{\mathbf{x}}_s
+
(\alpha_t-\alpha_s)
{\mathbf{x}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s)
+
(\sigma_t-\sigma_s)
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s).
$$

因此，它们精确的有限步差异为

$$
\begin{aligned}
\tilde{\mathbf{x}}_t^{\mathrm{DDIM}}
-
\tilde{\mathbf{x}}_t^{\mathrm{Euler}}
={}&
\left[
\alpha_t-\alpha_s+\Delta s\,\alpha_s'
\right]
{\mathbf{x}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s)
+
\left[
\sigma_t-\sigma_s+\Delta s\,\sigma_s'
\right]
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s).
\end{aligned}
$$

对于光滑调度，

$$
\alpha_t-\alpha_s+\Delta s\,\alpha_s'
=
\tfrac{(\Delta s)^2}{2}\alpha_s''
+
\mathcal O((\Delta s)^3), \quad
\sigma_t-\sigma_s+\Delta s\,\sigma_s'
=
\tfrac{(\Delta s)^2}{2}\sigma_s''
+
\mathcal O((\Delta s)^3).
$$

于是，

$$
\begin{aligned}
\tilde{\mathbf{x}}_t^{\mathrm{DDIM}}
-
\tilde{\mathbf{x}}_t^{\mathrm{Euler}}
=
\tfrac{(\Delta s)^2}{2}
\left[
\alpha_s''
{\mathbf{x}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s)
+
\sigma_s''
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s)
\right]
+
\mathcal O((\Delta s)^3).
\end{aligned}
$$

因此，DDIM 与普通 Euler 在一阶上一致，但有限步更新一般不同。普通 Euler 用局部线性化替代有限调度变化 $(\alpha_t-\alpha_s,\sigma_t-\sigma_s)$，而 DDIM 解析保留有限调度演化。

可以立即得到一个有用特例。如果 $\alpha_\tau$ 和 $\sigma_\tau$ 都*关于 $\tau$ 在 $[t,s]$ 上仿射*，则

$$
\alpha_t-\alpha_s=-\Delta s\,\alpha_s',
\qquad
\sigma_t-\sigma_s=-\Delta s\,\sigma_s',
$$

所以 DDIM 与普通 Euler 完全一致。例如，这包括以下调度（另见[第 6.3.4 节](../06-unified-perspective/03-equivalent-parameterizations.md#subsec-concept-why-v-affine)的讨论）

$$
(\alpha_t,\sigma_t)=(1,t)\quad \text{and}\quad (\alpha_t,\sigma_t)=(1-t,t) .
$$

因此，两种方法有限步上的差别，来自已知调度相对于原始扩散时间的非线性。

**DDIM 与指数 Euler。** 上面的比较使用干净数据／噪声表示，展示 DDIM 与原始扩散时间下普通 Euler 的差异。现在回到引言中的半线性表示，以区分 DDIM 与通常的指数 Euler。区别很简单：两种方法都解析保留已知线性演化，但在剩余积分内固定的量不同。

在噪声预测下，[式 (9.1.4)](01-prologue.md#eq-general-semilinear-pfode)中的因子分解 PF-ODE 具有

$$
a(\tau)=f(\tau),
\qquad
b(\tau)=\frac{g^2(\tau)}{2\sigma_\tau},
\qquad
{\mathbf{N}}_{\boldsymbol{\phi}^\times}
=
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}.
$$

其精确常数变易公式包含模型相关项

$$
b(\tau){\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau).
$$

通常的指数 Euler 将这一系数加权量视为单个残差，并将整个残差固定为该步起点的值：

$$
b(\tau){\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)
\approx
b(s){\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s), \quad\tau\in[t,s]
$$

因此得到

$$
\tilde{\mathbf{x}}_t^{\mathrm{ExpEuler}}
=
\mathcal E(s\shortrightarrow t)\tilde{\mathbf{x}}_s
+
b(s)
\left[
\int_s^t
\mathcal E(\tau\shortrightarrow t)
\mathop{}\!\mathrm{d}\tau
\right]
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s).
$$

DDIM 使用引言中强调的更细致分解。由于 $b(\tau)$ 由扩散调度确定，仍保留在积分内；只固定神经网络预测：

$$
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)
\approx
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s).
$$

于是，

$$
\tilde{\mathbf{x}}_t^{\mathrm{DDIM}}
=
\mathcal E(s\shortrightarrow t)\tilde{\mathbf{x}}_s
+
\left[
\int_s^t
\mathcal E(\tau\shortrightarrow t)b(\tau)
\mathop{}\!\mathrm{d}\tau
\right]
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s).
$$

因此，两种更新的有限步差异完全来自已知系数 $b(\tau)$：

$$
\tilde{\mathbf{x}}_t^{\mathrm{DDIM}}
-
\tilde{\mathbf{x}}_t^{\mathrm{ExpEuler}}
=
\left[
\int_s^t
\mathcal E(\tau\shortrightarrow t)
\bigl(b(\tau)-b(s)\bigr)
\mathop{}\!\mathrm{d}\tau
\right]
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s).
$$

对于光滑的 $b$，

$$
\tilde{\mathbf{x}}_t^{\mathrm{DDIM}}
-
\tilde{\mathbf{x}}_t^{\mathrm{ExpEuler}}
=
\frac{(\Delta s)^2}{2}
b'(s)
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s)
+
\mathcal O((\Delta s)^3).
$$

因此，DDIM 与通常的指数 Euler 在一阶上一致。唯一区别在于，指数 Euler 将 $b(\tau)$ 固定在 $b(s)$，与神经网络预测一同冻结；DDIM 则保留 $b(\tau)$ 的已知变化。所以，只要 $b(\tau)$ 在该步内恒定，两种更新就完全一致，包括引言示例使用的常数 $b$ 情形。

结合前面的普通 Euler 比较，三种一阶规则的区别，仅在于对 $\tau\in[t,s]$ 作常数近似之前保留多少已知结构：

$$
\begin{array}{rcl}
\text{Plain Euler}
&:&
f(\tau){\mathbf{x}}_\tau+b(\tau){\boldsymbol{\epsilon}}_\tau
\;\approx\;
f(s)\tilde{\mathbf{x}}_s+b(s)\widehat{\boldsymbol{\epsilon}},
\\[4pt]
\text{Exponential Euler}
&:&
b(\tau){\boldsymbol{\epsilon}}_\tau
\;\approx\;
b(s)\widehat{\boldsymbol{\epsilon}},
\\[4pt]
\text{DDIM}
&:&
{\boldsymbol{\epsilon}}_\tau
\;\approx\;
\widehat{\boldsymbol{\epsilon}},
\end{array}
$$

其中 $\widehat{\boldsymbol{\epsilon}} := {\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s)$。三者都使用该步起点处的同一次神经网络计算。普通 Euler 近似完整速度；指数 Euler 先保留已知线性传播；DDIM 还进一步保留已知系数 $b(\tau)$，只近似神经网络预测。

这一层次关系描述保留哪些解析已知项，并不意味着单步误差存在普遍的大小排序。其主要数值意义在于，DDIM 的剩余近似可以完全归结为神经网络预测的变化，下文将作说明。

**每种方法仍会产生什么误差？** 上面的比较描述了三种数值更新彼此如何不同。现在将每种更新与精确经验 PF-ODE 流比较。进行局部误差分析时，假设该步从精确经验状态 ${\mathbf{x}}_s$ 出发，其时刻为 $s$。

对于普通 Euler，精确经验流可以写成直接速度形式

$$
\widetilde{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
=
{\mathbf{x}}_s
+
\int_s^t
{\mathbf{v}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)
\mathop{}\!\mathrm{d}\tau.
$$

由于普通 Euler 将完整速度固定为起始时刻的值，其精确单步误差为

$$
\begin{aligned}
\widetilde{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
-
\tilde{\mathbf{x}}_t^{\mathrm{Euler}}
=
\int_s^t
\Big[
{\mathbf{v}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)
-
{\mathbf{v}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_s,s)
\Big]
\mathop{}\!\mathrm{d}\tau.
\end{aligned}
$$

对于指数 Euler，已知线性传播已被解析处理，但系数加权残差 $b(\tau){\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)$ 保持固定。因此

$$
\begin{aligned}
\widetilde{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
-
\tilde{\mathbf{x}}_t^{\mathrm{ExpEuler}}
=
\int_s^t
\mathcal E(\tau\shortrightarrow t)
\Big[
b(\tau)
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)
-
b(s)
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_s,s)
\Big]
\mathop{}\!\mathrm{d}\tau.
\end{aligned}
$$

最后，DDIM 同时保留已知线性传播及已知系数 $b(\tau)$，只固定神经网络预测：

<span id="eq-ddim-local-defect"></span>

$$
\begin{aligned}
\widetilde{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
-
\tilde{\mathbf{x}}_t^{\mathrm{DDIM}}
=
\int_s^t
\mathcal E(\tau\shortrightarrow t)b(\tau)
\Big[
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)
-
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_s,s)
\Big]
\mathop{}\!\mathrm{d}\tau.
\end{aligned}
\tag{9.2.3}
$$

这些恒等式揭示了各一阶格式相关的光滑性概念。如果相应量沿精确经验轨迹缓慢变化，则单步近似准确。更精确地，取 $\Delta s:=s-t$，标准光滑性条件给出

$$
\begin{aligned}
\left\|
\widetilde{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
-
\tilde{\mathbf{x}}_t^{\mathrm{Euler}}
\right\|
&\le
\frac{(\Delta s)^2}{2}
\sup_{\tau\in[t,s]}
\left\|
\tfrac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d}\tau}
{\mathbf{v}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)
\right\|,
\\
\left\|
\widetilde{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
-
\tilde{\mathbf{x}}_t^{\mathrm{ExpEuler}}
\right\|
&\le
\frac{(\Delta s)^2}{2}
\sup_{\tau\in[t,s]}
\left|
\mathcal E(\tau\shortrightarrow t)
\right|\cdot
\sup_{\tau\in[t,s]}
\left\|
\tfrac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d}\tau}
\left[
b(\tau)
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)
\right]
\right\|,
\\
\left\|
\widetilde{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
-
\tilde{\mathbf{x}}_t^{\mathrm{DDIM}}
\right\|
&\le
\frac{(\Delta s)^2}{2}
\sup_{\tau\in[t,s]}
\left|
\mathcal E(\tau\shortrightarrow t)b(\tau)
\right| \cdot
\sup_{\tau\in[t,s]}
\left\|
\tfrac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d}\tau}
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)
\right\|.
\end{aligned}
$$

在通常的光滑性与稳定性假设下，三种方法均为*全局一阶*，且每步只需一次神经网络计算。此外，当某种方法所固定的量沿精确经验轨迹实际上为常数时，该方法在这一步上精确：对普通 Euler 是完整 PF-ODE 速度，对指数 Euler 是系数加权残差 $b(\tau){\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)$，对 DDIM 是噪声预测 ${\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)$。

因此，三种方法的误差没有普遍的大小排序：当各自固定的量在一步内变化缓慢时，对应方法最准确。DDIM 特别适合神经网络预测缓慢变化、但已知扩散调度变化明显的情形，因为它解析保留了后者。因此，利用更多已知结构可以减少离散化误差，但不保证每一个步骤的误差都更小。

这里讨论的是相对于经验 PF-ODE 的数值离散化误差；经验 PF-ODE 与 oracle PF-ODE 之间的模型不匹配是另一种误差来源。

**DDIM 的精确 Euler 解释。** 尽管 DDIM 与普通 Euler 在一阶上一致，但在原始状态 ${\mathbf{x}}$ 和扩散时间 $t$ 下，一般给出不同的有限步更新。不过，对状态与时间坐标同时作与调度相适应的变换后，可以得到精确的 Euler 解释。

在噪声预测下，定义

$$
{\mathbf{y}}_t
:=
\frac{{\mathbf{x}}_t}{\alpha_t},
\qquad
\rho_t
:=
\frac{\sigma_t}{\alpha_t}.
$$

两种变换起到互补作用。状态归一化 ${\mathbf{x}}_t\mapsto{\mathbf{x}}_t/\alpha_t$ 提取出已知信号缩放 $\alpha_t$，而新时钟 $\rho_t=\sigma_t/\alpha_t$ 用噪声信号比而非原始扩散时间度量进展。

先在条件层面理解这一点，考虑 Gaussian 扰动路径中的固定样本对 $({\mathbf{x}}_0,{\boldsymbol{\epsilon}})$：

$$
{\mathbf{x}}_t
=
\alpha_t{\mathbf{x}}_0+\sigma_t{\boldsymbol{\epsilon}},
\qquad
{\boldsymbol{\epsilon}}\sim\mathcal N(\mathbf 0,\mathbf I).
$$

这里，${\boldsymbol{\epsilon}}$ 是固定的潜在噪声实现，并非神经网络预测。变换后的状态满足

$$
{\mathbf{y}}_t
=
\frac{{\mathbf{x}}_t}{\alpha_t}
=
{\mathbf{x}}_0+\rho_t{\boldsymbol{\epsilon}}.
$$

因此，沿这一特定条件路径，

$$
\frac{\mathop{}\!\mathrm{d}{\mathbf{y}}}{\mathop{}\!\mathrm{d}\rho}
=
{\boldsymbol{\epsilon}}.
$$

于是，对于固定 $({\mathbf{x}}_0,{\boldsymbol{\epsilon}})$，已知调度依赖在变换后的坐标 $\rho$ 中变为仿射形式。

相应经验 PF-ODE 具有相同坐标结构，但固定潜在噪声 ${\boldsymbol{\epsilon}}$ 被学习到的、依赖状态的噪声预测替代。利用[式 (9.1.5)](01-prologue.md#eq-noise-pf-ode)，

$$
\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}_t}{\mathop{}\!\mathrm{d} t}
=
\frac{\alpha_t'}{\alpha_t}{\mathbf{x}}_t
+
\left(
\sigma_t'
-
\frac{\alpha_t'}{\alpha_t}\sigma_t
\right)
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t,t),
$$

得到

<span id="eq-ddim-coordinate-ode-t"></span>

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d}{\mathbf{y}}_t}{\mathop{}\!\mathrm{d} t}
=
\frac{1}{\alpha_t}
\left(
\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}_t}{\mathop{}\!\mathrm{d} t}
-
\frac{\alpha_t'}{\alpha_t}{\mathbf{x}}_t
\right)=
\frac{\mathop{}\!\mathrm{d}\rho_t}{\mathop{}\!\mathrm{d} t}\,
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t,t).
\end{aligned}
\tag{9.2.4}
$$

因此，在满足 $\alpha_t>0$ 且 $\rho_t$ 严格单调的任何区间上，可以使用 $\rho$ 作为时间坐标。记 $t=t(\rho)$，并定义变换后的神经网络场

$$
\bar{{\boldsymbol{\epsilon}}}_{\boldsymbol{\phi}^\times}({\mathbf{y}},\rho)
:=
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}
\bigl(
\alpha_{t(\rho)}{\mathbf{y}},
t(\rho)
\bigr),
$$

经验 PF-ODE 变为

<span id="eq-ddim-coordinate-ode-rho"></span>

$$
\frac{\mathop{}\!\mathrm{d}{\mathbf{y}}}{\mathop{}\!\mathrm{d}\rho}
=
\bar{{\boldsymbol{\epsilon}}}_{\boldsymbol{\phi}^\times}({\mathbf{y}},\rho).
\tag{9.2.5}
$$

所以模型的空间与时间依赖依然存在，只是用变换后的变量 $({\mathbf{y}},\rho)$ 表达。

对[式 (9.2.5)](02-ddim.md#eq-ddim-coordinate-ode-rho)从 $\rho_s$ 到 $\rho_t$ 应用显式 Euler，得到

$$
\tilde{\mathbf{y}}_t
=
\tilde{\mathbf{y}}_s
+
(\rho_t-\rho_s)
\bar{{\boldsymbol{\epsilon}}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{y}}_s,\rho_s).
$$

由于

$$
\bar{{\boldsymbol{\epsilon}}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{y}}_s,\rho_s)
=
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s),
$$

代入 ${\mathbf{y}}={\mathbf{x}}/\alpha$ 与 $\rho=\sigma/\alpha$，得到

$$
\tilde{\mathbf{x}}_t
=
\frac{\alpha_t}{\alpha_s}\tilde{\mathbf{x}}_s
+
\alpha_t
\left(
\frac{\sigma_t}{\alpha_t}
-
\frac{\sigma_s}{\alpha_s}
\right)
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s),
$$

这恰好是确定性 DDIM 更新。

**坐标选择是数值方法的一部分**

可逆的状态与时间坐标变换，为同一连续 ODE 解提供等价表示。但有限步离散化一般不对这种表示保持不变：“先变换坐标再应用 Euler”，不一定等于“先应用 Euler 再变换坐标”。

其数值意义在于，合适的坐标系可以在离散化前吸收解析已知的动力学部分。DDIM 中，${\mathbf{x}}/\alpha$ 去除已知信号缩放，$\rho=\sigma/\alpha$ 吸收相应调度相关时间缩放，留下变换后的神经网络场 $\bar{{\boldsymbol{\epsilon}}}_{\boldsymbol{\phi}^\times}({\mathbf{y}},\rho)$ 作为 Euler 近似的对象。因此，DDIM 通常不同于 $({\mathbf{x}},t)$ 中的普通 Euler，但它恰好是调度适配坐标 $({\mathbf{y}},\rho)$ 中的显式 Euler。

干净数据预测下有一个对偶构造。用 $\sigma_t$ 归一化状态，以 $\alpha_t/\sigma_t$ 为时间坐标，得到

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} \left( \frac{{\mathbf{x}}_t}{\sigma_t} \right) = \frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} \left( \frac{\alpha_t}{\sigma_t} \right) {\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t,t),
$$

因此，DDIM 的干净数据形式同样是在相应调度适配坐标中的显式 Euler。

这一*状态与时间坐标变换视角*也预示了坐标选择在高阶扩散求解器中的作用。后文[第 9.4 节](04-dpm-solver.md#sec-dpm)介绍的 DPM-Solver 使用半对数信噪比

$$
\lambda_t
:=
\log\frac{\alpha_t}{\sigma_t},
$$

它与 DDIM 的 Euler 时钟直接相关：

$$
\rho_t
=
\frac{\sigma_t}{\alpha_t}
=
e^{-\lambda_t},
\qquad
\mathop{}\!\mathrm{d}\rho
=
-e^{-\lambda}\mathop{}\!\mathrm{d}\lambda.
$$

因此，精确的变换后方程

$$
\frac{\mathop{}\!\mathrm{d}{\mathbf{y}}}{\mathop{}\!\mathrm{d}\rho}
=
\bar{{\boldsymbol{\epsilon}}}_{\boldsymbol{\phi}^\times}({\mathbf{y}},\rho)
$$

也可以等价地按如下方式积分

$$
{\mathbf{y}}_t
=
{\mathbf{y}}_s
-
\int_{\lambda_s}^{\lambda_t}
e^{-\lambda}
\hat{{\boldsymbol{\epsilon}}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda)
\mathop{}\!\mathrm{d}\lambda,
$$

其中采用通常的 $\lambda$ 参数化模型记号，该记号将在[第 9.4 节](04-dpm-solver.md#sec-dpm)介绍。

在一阶情况下，固定神经网络预测并保留已知变量替换权重，会给出相同的有限步更新，因此 DPM-Solver-1 恰好等于 DDIM。但在更高阶情况下，选择在哪个坐标中近似依赖状态与时间的神经网络预测，就成为数值方法的一部分。DPM-Solver 使用 $\lambda$，因为它会得到便于进行高阶展开的指数加权积分。

## 9.2.3 （选读）DDIM 最初的变分视角 {#subsec-ddim}

上述 PF-ODE 视角将确定性 DDIM 解释为数值求解器。不过，DDIM 最初是从 DDPM 的变分视角提出的。其出发问题不同：*去噪目标是否唯一决定了不同时刻的带噪状态必须如何耦合？*

**回顾 DDPM 的变分视角。** 回顾 DDPM，它引入固定的 Markov 推断过程，其单时刻扰动边缘分布具有 Gaussian 形式

$$
p_t({\mathbf{x}}_t|{\mathbf{x}}_0)
=
\mathcal N
\bigl(
{\mathbf{x}}_t;
\alpha_t{\mathbf{x}}_0,
\sigma_t^2{\mathbf{I}}
\bigr).
$$

相应的变分推导涉及可计算的条件分布，如 $p({\mathbf{x}}_t|{\mathbf{x}}_s,{\mathbf{x}}_0)$，其中 $t<s$；而用于生成的学习逆向模型无法直接以未知干净样本 ${\mathbf{x}}_0$ 为条件，必须依赖扩散模型预测。

DDIM 背后的关键观察与去噪目标有关。回顾噪声预测 DDPM 损失，它可写为

$$
{\mathbb{E}}_t
{\mathbb{E}}_{{\mathbf{x}}_0,{\boldsymbol{\epsilon}}}
\left[
\left\|
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}}
\bigl(
\alpha_t{\mathbf{x}}_0+\sigma_t{\boldsymbol{\epsilon}},t
\bigr)
-
{\boldsymbol{\epsilon}}
\right\|_2^2
\right],
$$

该损失通过单时刻边缘分布 $p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$ 依赖于推断过程，并不通过 ${\mathbf{x}}_1,\ldots,{\mathbf{x}}_T$ 在给定 ${\mathbf{x}}_0$ 时的完整联合耦合来依赖它。

**DDIM 最初的动机。** DDIM 的核心观察是：去噪训练目标不唯一决定不同时刻的带噪状态如何耦合。对于固定干净样本 ${\mathbf{x}}_0$，通常的去噪代理目标由单时刻扰动边缘分布 $p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$ 决定，而许多不同的跨时间联合过程可以具有完全相同的边缘分布。DDIM 利用这一自由度：保留训练使用的扰动边缘分布，同时改变不同噪声水平之间的耦合。

具体而言，对于任意 $t<s$，考虑条件转移 ${\color{orange}\pi({\mathbf{x}}_t|{\mathbf{x}}_s,{\mathbf{x}}_0)}$。要求它具有*单步边缘一致性*：

<span id="eq-ddim-reverse-kernel-def"></span>

$$
\begin{aligned}
\int
{\color{orange}\pi({\mathbf{x}}_t|{\mathbf{x}}_s,{\mathbf{x}}_0)}
\,p_s({\mathbf{x}}_s|{\mathbf{x}}_0)
\mathop{}\!\mathrm{d}{\mathbf{x}}_s
=
p_t({\mathbf{x}}_t|{\mathbf{x}}_0).
\end{aligned}
\tag{9.2.6}
$$

也就是说，如果 ${\mathbf{x}}_s\sim p_s(\cdot|{\mathbf{x}}_0)$，那么应用一次选定转移就得到 ${\mathbf{x}}_t\sim p_t(\cdot|{\mathbf{x}}_0)$。因此，即使 ${\mathbf{x}}_s$ 与 ${\mathbf{x}}_t$ 之间的依赖关系改变，每个噪声水平的边缘分布仍然保留。

原始 Markov DDPM 前向过程诱导的 Bayes 逆向后验，是这类转移的一个有效选择[^2]，但边缘分布并不唯一决定它。DDIM 允许其他边缘一致的耦合，包括在相隔很远的噪声水平之间直接转移。这就是为何同一训练好的去噪器可以复用于更粗的采样网格、跳过中间时刻的概率解释。

**离散时间 DDIM 的推导。** 现在对于 Gaussian 扰动模型，显式写出上述耦合自由度

$$
p_t({\mathbf{x}}_t|{\mathbf{x}}_0)
:=
\mathcal{N}
\bigl(
{\mathbf{x}}_t;
\alpha_t{\mathbf{x}}_0,
\sigma_t^2\mathbf{I}
\bigr),
\qquad
{\mathbf{x}}_0\sim p_{\mathrm{data}}.
$$

目标是对于任意 $t<s$，构造满足[式 (9.2.6)](02-ddim.md#eq-ddim-reverse-kernel-def)中边缘一致性条件的条件转移 ${\color{orange}\pi({\mathbf{x}}_t|{\mathbf{x}}_s,{\mathbf{x}}_0)}$，而不要求它等于原始 Markov 前向过程对应的 Bayes 后验。

一个自然选择是 Gaussian 族

<span id="eq-ddim-reverse-kernel-param"></span>

$$
\begin{aligned}
{\color{orange}\pi({\mathbf{x}}_t|{\mathbf{x}}_s,{\mathbf{x}}_0)}
=
\mathcal{N}
\bigl(
{\mathbf{x}}_t;\,
a_{t,s}{\mathbf{x}}_0+b_{t,s}{\mathbf{x}}_s,\,
c_{t,s}^2\mathbf{I}
\bigr),
\end{aligned}
\tag{9.2.7}
$$

其中，$(a_{t,s},b_{t,s},c_{t,s})$ 决定时刻 $s$ 的状态、干净样本及新注入噪声如何组合。

为施加边缘一致性，先采样

$$
{\mathbf{x}}_s
=
\alpha_s{\mathbf{x}}_0+\sigma_s{\boldsymbol{\epsilon}}',
\qquad
{\boldsymbol{\epsilon}}'\sim\mathcal{N}(\mathbf{0},\mathbf{I}),
$$

再从[式 (9.2.7)](02-ddim.md#eq-ddim-reverse-kernel-param)采样 ${\mathbf{x}}_t$。等价地，

<span id="eq-ddim-derivation"></span>

$$
\begin{aligned}
{\mathbf{x}}_t
&=
a_{t,s}{\mathbf{x}}_0
+
b_{t,s}{\mathbf{x}}_s
+
c_{t,s}{\boldsymbol{\epsilon}}
\\
&=
a_{t,s}{\mathbf{x}}_0
+
b_{t,s}
\bigl(
\alpha_s{\mathbf{x}}_0+\sigma_s{\boldsymbol{\epsilon}}'
\bigr)
+
c_{t,s}{\boldsymbol{\epsilon}}
\\
&=
\bigl(
a_{t,s}+b_{t,s}\alpha_s
\bigr){\mathbf{x}}_0
+
\sqrt{
b_{t,s}^2\sigma_s^2+c_{t,s}^2
}\,
{\boldsymbol{\epsilon}}'',
\end{aligned}
\tag{9.2.8}
$$

其中，${\boldsymbol{\epsilon}}$ 与 ${\boldsymbol{\epsilon}}'$ 是独立标准 Gaussian 变量，${\boldsymbol{\epsilon}}''$ 是表示其归一化线性组合的标准 Gaussian 变量。

要使所得边缘分布等于给定扰动边缘分布 ${\mathbf{x}}_t \sim p_t({\mathbf{x}}_t|{\mathbf{x}}_0) = \mathcal{N} \bigl( {\mathbf{x}}_t; \alpha_t{\mathbf{x}}_0, \sigma_t^2\mathbf{I} \bigr)$，均值和方差必须一致。因此，

$$
\begin{aligned}
\alpha_t
&=
a_{t,s}+b_{t,s}\alpha_s,
\\
\sigma_t^2
&=
b_{t,s}^2\sigma_s^2+c_{t,s}^2.
\end{aligned}
$$

关键在于，这只是对三个系数 $(a_{t,s},b_{t,s},c_{t,s})$ 的两个约束。因此，给定边缘分布*不能*唯一确定 $s$ 与 $t$ 之间的转移。可以将 $c_{t,s}$ 选为自由参数，满足 $0\leq c_{t,s}\leq\sigma_t$，再求解其余系数。选择非负分支，得到

<span id="eq-ddim-coefficients"></span>

$$
\begin{aligned}
b_{t,s}
&=
\frac{
\sqrt{\sigma_t^2-c_{t,s}^2}
}{
\sigma_s
},
\qquad
a_{t,s}
=
\alpha_t-\alpha_s b_{t,s}.
\end{aligned}
\tag{9.2.9}
$$

对 $b_{t,s}$ 取非负值，就得到标准 DDIM 耦合。

将[式 (9.2.9)](02-ddim.md#eq-ddim-coefficients)代入[式 (9.2.7)](02-ddim.md#eq-ddim-reverse-kernel-param)，得到

<span id="eq-ddim-reverse-kernel-result"></span>

$$
\begin{aligned}
{\color{orange}\pi({\mathbf{x}}_t|{\mathbf{x}}_s,{\mathbf{x}}_0)}
=
\mathcal{N}
\left(
{\mathbf{x}}_t;\,
\alpha_t{\mathbf{x}}_0
+
\frac{
\sqrt{\sigma_t^2-c_{t,s}^2}
}{
\sigma_s
}
\bigl(
{\mathbf{x}}_s-\alpha_s{\mathbf{x}}_0
\bigr),
\,
c_{t,s}^2\mathbf{I}
\right).
\end{aligned}
\tag{9.2.10}
$$

这个表达式显式展示了耦合自由度。${\mathbf{x}}_s-\alpha_s{\mathbf{x}}_0$ 是 ${\mathbf{x}}_s$ 中已经存在的噪声分量，$c_{t,s}$ 则控制从 $s$ 到 $t$ 的转移中引入多少*新的* Gaussian 噪声。因此，改变 $c_{t,s}$ 会改变两个噪声水平之间的耦合，同时保留相同的目标边缘分布 $p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$。

**说明。**

选定转移 ${\color{orange}\pi({\mathbf{x}}_t|{\mathbf{x}}_s,{\mathbf{x}}_0)}$ 一般不同于原始 Markov 前向过程对应的 Bayes 后验。通过对 $c_{t,s}$ 作某个特定选择，可以恢复该后验，下文将说明。其他选择定义不同的边缘一致耦合，包括在 $c_{t,s}=0$ 时得到的确定性 DDIM 耦合。

**DDIM 采样器（步骤 $s\to t$）。** [式 (9.2.10)](02-ddim.md#eq-ddim-reverse-kernel-result)中的边缘一致核 ${\color{orange}\pi({\mathbf{x}}_t|{\mathbf{x}}_s,{\mathbf{x}}_0)}$ 仍以未知干净样本 ${\mathbf{x}}_0$ 为条件。生成时，DDIM 将 ${\mathbf{x}}_0$ 替换为预训练扩散模型给出的干净数据估计。在 ${\boldsymbol{\epsilon}}$ 预测下，定义

$$
{\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_s,s)
:=
\frac{
{\mathbf{x}}_s
-
\sigma_s{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_s,s)
}{
\alpha_s
},
$$

并使用代入后的转移

$$
p_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t|{\mathbf{x}}_s)
:=
{\color{orange}\pi
\bigl(
{\mathbf{x}}_t
\,\big|\,
{\mathbf{x}}_s,
{\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_s,s)
\bigr)}.
$$

将此估计代入[式 (9.2.10)](02-ddim.md#eq-ddim-reverse-kernel-result)，得到

$$
{\mathbf{x}}_t
=
\frac{\alpha_t}{\alpha_s}{\mathbf{x}}_s
+
\left(
\sqrt{\sigma_t^2-c_{t,s}^2}
-
\frac{\alpha_t}{\alpha_s}\sigma_s
\right)
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_s,s)
+
c_{t,s}{\boldsymbol{\epsilon}}_t,
\qquad
{\boldsymbol{\epsilon}}_t\sim\mathcal{N}(\mathbf{0},\mathbf{I}).
$$

因此，模型预测决定从 $s$ 携带到 $t$ 的干净数据／噪声估计内容，自由系数 $c_{t,s}\in[0,\sigma_t]$ 则控制该步中注入多少*新的* Gaussian 噪声。

为解释这一自由度，可以与原始 Markov 前向过程比较。对于 $t<s$，定义

$$
\alpha_{s|t}
:=
\frac{\alpha_s}{\alpha_t},
\qquad
\sigma_{s|t}^2
:=
\sigma_s^2-\alpha_{s|t}^2\sigma_t^2,
$$

使得

$$
p({\mathbf{x}}_s|{\mathbf{x}}_t)
=
\mathcal{N}
\bigl(
{\mathbf{x}}_s;
\alpha_{s|t}{\mathbf{x}}_t,
\sigma_{s|t}^2\mathbf I
\bigr).
$$

Gaussian 条件化给出相应后验方差

$$
\operatorname{Var}
\bigl(
{\mathbf{x}}_t|{\mathbf{x}}_s,{\mathbf{x}}_0
\bigr)
=
\frac{\sigma_t^2}{\sigma_s^2}
\sigma_{s|t}^2\mathbf I.
$$

这将 DDPM 后验确定为上述边缘一致族中的一个特定成员，并明确了 $c_{t,s}$ 的作用：

-   **DDPM 步（后验方差）。** 选择

    $$
    c_{t,s}
    =
    \frac{\sigma_t}{\sigma_s}\sigma_{s|t}
    $$

    就匹配原始 Markov 前向过程的后验方差。结合[式 (9.2.9)](02-ddim.md#eq-ddim-coefficients)中的非负分支，条件核 ${\color{orange}\pi({\mathbf{x}}_t|{\mathbf{x}}_s,{\mathbf{x}}_0)}$ 此时与该过程的 Bayes 后验 $p({\mathbf{x}}_t|{\mathbf{x}}_s,{\mathbf{x}}_0)$ 一致。

-   **确定性 DDIM（$\eta=0$）。** 在另一个极端，令 $c_{t,s}=0$，就去掉所有新注入噪声。实际更新变为

    $$
    {\mathbf{x}}_t
    =
    \frac{\alpha_t}{\alpha_s}{\mathbf{x}}_s
    +
    \left(
    \sigma_t
    -
    \frac{\alpha_t}{\alpha_s}\sigma_s
    \right)
    {\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_s,s),
    $$

    这恰好是前面从 PF-ODE 推导出的确定性 DDIM 更新。

    在条件层面，它的共享噪声解释特别简单：如果 ${\mathbf{x}}_s=\alpha_s{\mathbf{x}}_0+\sigma_s{\boldsymbol{\epsilon}}$，则确定性转移给出

    $$
    {\mathbf{x}}_t
    =
    \alpha_t{\mathbf{x}}_0+\sigma_t{\boldsymbol{\epsilon}}.
    $$

    因此，没有引入新噪声；同一样本对 $({\mathbf{x}}_0,{\boldsymbol{\epsilon}})$ 贯穿这一采样步，只有已知调度系数发生变化。

-   **插值。** 更一般地，定义

    <span id="eq-eta-ddim"></span>

    $$
    \begin{aligned}
    c_{t,s}
    =
    \eta\,
    \frac{\sigma_t}{\sigma_s}\,
    \sigma_{s|t},
    \qquad
    \eta\in[0,1].
    \end{aligned}
    \tag{9.2.11}
    $$

    参数 $\eta$ 连续控制转移的随机性：$\eta=0$ 给出确定性 DDIM，$\eta=1$ 则匹配相应 DDPM Markov 转移的后验方差。

因此，DDIM 族将两个以不同方式固定的要素分开：扰动边缘分布保持不变，而相邻噪声水平之间的耦合可以通过 $c_{t,s}$ 改变。确定性 DDIM 对应于在该步内保留已有共享噪声，而 DDPM 后验选择则注入原始 Markov 前向过程所规定的额外随机性。

**与条件流匹配的联系。** 上述确定性 DDIM 构造，还与[第 5.2 节](../05-flow-based-perspective/02-flow-matching-framework.md#sec-flow-matching-framework)中的条件流匹配（CFM）存在简单联系。关键是区分同一条件 Gaussian 路径对应的两个对象：*有限时间映射*与*瞬时速度*。DDIM 与前者相关，CFM 则基于后者构建。

给定干净样本 ${\mathbf{x}}_0$，考虑共享噪声路径

$$
{\mathbf{x}}_\tau
=
\alpha_\tau{\mathbf{x}}_0
+
\sigma_\tau{\boldsymbol{\epsilon}}.
$$

对于固定 $({\mathbf{x}}_0,{\boldsymbol{\epsilon}})$，这是穿过 Gaussian 扰动边缘分布 $p_\tau({\mathbf{x}}_\tau|{\mathbf{x}}_0)$ 的精确条件轨迹。

**有限时间视角：条件 DDIM 映射。** 假设轨迹位于 ${\mathbf{x}}_s$，时刻为 $s$。由于保持同一个 ${\boldsymbol{\epsilon}}$，从 ${\mathbf{x}}_s=\alpha_s{\mathbf{x}}_0+\sigma_s{\boldsymbol{\epsilon}}$ 中消去它，得到精确条件映射

$$
{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s|{\mathbf{x}}_0)
=
\frac{\sigma_t}{\sigma_s}{\mathbf{x}}_s
+
\left(
\alpha_t
-
\alpha_s\frac{\sigma_t}{\sigma_s}
\right){\mathbf{x}}_0.
$$

该映射*在给定 ${\mathbf{x}}_0$ 的条件下*精确。它还不是实际 DDIM 采样器，因为生成时 ${\mathbf{x}}_0$ 未知。将 ${\mathbf{x}}_0$ 替换为起始时刻的模型预测，就恰好得到上面推导的确定性 DDIM 更新。

**无穷小时间视角：CFM 条件速度。** 除了询问条件路径在有限区间内移到哪里，也可以询问其瞬时方向。对 ${\mathbf{x}}_t=\alpha_t{\mathbf{x}}_0+\sigma_t{\boldsymbol{\epsilon}}$ 求导，得到

$$
\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}_t}{\mathop{}\!\mathrm{d} t}
=
\alpha_t'{\mathbf{x}}_0+\sigma_t'{\boldsymbol{\epsilon}}.
$$

利用

$$
{\boldsymbol{\epsilon}}
=
\frac{{\mathbf{x}}-\alpha_t{\mathbf{x}}_0}{\sigma_t},
$$

可以将同一速度写成当前状态的函数：

$$
{\mathbf{v}}_t^*({\mathbf{x}}|{\mathbf{x}}_0)
=
\frac{\sigma_t'}{\sigma_t}{\mathbf{x}}
+
\left(
\alpha_t'
-
\alpha_t\frac{\sigma_t'}{\sigma_t}
\right){\mathbf{x}}_0.
$$

这就是条件流匹配中 Gaussian 概率路径对应的条件速度。

随后，CFM 在当前状态下，对关于 ${\mathbf{x}}_0$ 的后验不确定性取平均，将条件速度联系到边缘动力学：

$$
\begin{aligned}
{\mathbf{v}}^*({\mathbf{x}},t)
&:=
{\mathbb{E}}_{{\mathbf{x}}_0\sim p({\mathbf{x}}_0|{\mathbf{x}}_t={\mathbf{x}})}
\left[
{\mathbf{v}}_t^*({\mathbf{x}}|{\mathbf{x}}_0)
\right]
\\
&=
\frac{\sigma_t'}{\sigma_t}{\mathbf{x}}
+
\left(
\alpha_t'
-
\alpha_t\frac{\sigma_t'}{\sigma_t}
\right)
{\mathbf{x}}^*({\mathbf{x}},t)
\\
&=
\alpha_t'{\mathbf{x}}^*({\mathbf{x}},t)
+
\sigma_t'{\boldsymbol{\epsilon}}^*({\mathbf{x}},t),
\end{aligned}
$$

其中使用了

$$
{\mathbf{x}}^*({\mathbf{x}},t)
=
{\mathbb{E}}[{\mathbf{x}}_0|{\mathbf{x}}_t={\mathbf{x}}],
\qquad
{\mathbf{x}}
=
\alpha_t{\mathbf{x}}^*({\mathbf{x}},t)
+
\sigma_t{\boldsymbol{\epsilon}}^*({\mathbf{x}},t).
$$

所得 ${\mathbf{v}}^*({\mathbf{x}},t)$ 恰好是 oracle 边缘 PF-ODE 速度。

**DDIM 与 CFM**

同一条件 Gaussian 路径提供两个视角：DDIM 将未知 ${\mathbf{x}}_0$ 替换为模型预测后，使用有限时间映射；CFM 则使用瞬时条件速度。对后者取后验平均，就得到边缘 PF-ODE 速度。

这一联系在条件路径及其瞬时速度层面是精确的。它*不*意味着有限 DDIM 步就是精确边缘 PF-ODE 流：DDIM 固定该步起点推断出的预测，而精确边缘 PF-ODE 速度会随演化状态连续变化。

[^1]: 本章中，$\tilde{{\mathbf{x}}}_s$ 表示时间层级 $s$ 的状态，用于初始化当前求解器步骤，通常来自前一个数值步骤。波浪号将其与 ${\mathbf{x}}_s$ 区分，后者表示精确扩散分布中的 oracle 样本，分布可以是 ${\mathbf{x}}_s \sim p_s$，也可以在给定 ${\mathbf{x}}_0$ 时为 ${\mathbf{x}}_s \sim p_s(\,\cdot \mid {\mathbf{x}}_0)$。因此，$\tilde{{\mathbf{x}}}_s$ 不必精确服从 $p_s$。

[^2]: 对于 Markov 前向过程，${\mathbf{x}}_0\to{\mathbf{x}}_t\to{\mathbf{x}}_s$，其中 $t<s$，条件后验 $p({\mathbf{x}}_t|{\mathbf{x}}_s,{\mathbf{x}}_0)$ 满足

    $$
    p_t({\mathbf{x}}_t|{\mathbf{x}}_0)
    =
    \int
    p({\mathbf{x}}_t|{\mathbf{x}}_s,{\mathbf{x}}_0)
    \,p_s({\mathbf{x}}_s|{\mathbf{x}}_0)
    \mathop{}\!\mathrm{d}{\mathbf{x}}_s
    $$

    这是全概率公式的结果。
