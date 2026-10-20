# 9.1 引言 {#sec-prologue-solvers}

## 9.1.1 快速采样作为数值积分 {#fast-sampling-as-numerical-integration}

Score SDE 框架（[Song et al., 2020](../../references.md#ref-song2020score)）将离散时间扩散模型及基于 ELBO 的表述（[Sohl-Dickstein et al., 2015](../../references.md#ref-sohl2015deep); [Ho et al., 2020](../../references.md#ref-ho2020denoising)）与连续时间 SDE／ODE 生成建模联系起来，奠定了关键基础。这一联系为理解采样提供了清晰方式：扩散模型训练完成后，可以将生成视为求解学习到的微分方程，从噪声返回数据。

实际困难在于速度。简单采样器通过许多数值小步沿逆时间轨迹前进，往往需要数百或数千次神经网络计算。快速采样研究能否以少得多的模型调用来近似同一个生成过程。

本章关注*免训练*加速：保持预训练扩散模型固定，设计更好的数值求解器用于采样。蒸馏或直接少步训练等基于训练的加速方法，将在[第 10 章](../10-distillation/index.md#ch-distillation)和[第 11 章](../11-fast-generators/index.md#ch-fast-scratch)讨论。

因此，本章要传达的主要观点很简单：

**要点**

扩散模型训练完成后，免训练快速采样在很大程度上就是数值积分问题。

文献中有许多各具名称的扩散采样器，容易让这一主题显得比实际更零散。本章先介绍支撑这些方法的少数数值思想，再将熟悉的方法名称与它们对应起来。这种组织方式说明，这些采样器具有共同基础：它们是对同一 PF-ODE 轨迹进行近似的不同方式。

**采样作为经验 PF-ODE 的求解。** 假设已有预训练扩散模型

$$
{\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}},t)
    \approx
    \nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}}),
$$

或等价地，使用[第 6.3 节](../06-unified-perspective/03-equivalent-parameterizations.md#sec-equivalent-parametrizations)中的其他标准参数化表达的模型。经验 PF-ODE 用学习到的模型替代真实 score：

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t}
    =
    {\mathbf{f}}({\mathbf{x}}(t),t)
    -
    \frac{1}{2}g^2(t)\,
    {\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}(t),t).
$$

从 ${\mathbf{x}}(T)\sim p_{\mathrm{prior}}$ 出发，生成过程将这一 ODE 从 $t=T$ 向后积分到 $t=0$。

对于两个时刻 $s>t$，精确的经验更新可以写为积分形式

<span id="eq-solution-int-score"></span>

$$
\begin{aligned}
    \widetilde{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
    =
    {\mathbf{x}}_s
    +
    \int_s^t
    \left[
        {\mathbf{f}}({\mathbf{x}}(\tau),\tau)
        -
        \frac{1}{2}g^2(\tau)
        {\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}(\tau),\tau)
    \right]\mathop{}\!\mathrm{d} \tau .
\end{aligned}
\tag{9.1.1}
$$

这里，$\widetilde{{\boldsymbol{\Psi}}}_{s\to t}$ 表示经验 PF-ODE 的流映射，由学习模型替代 oracle score 后得到。与全书其他部分一样，*流映射*指[式 (4.2.2)](../04-score-sde-framework/02-matching-marginals.md#eq-flow-map-def)介绍的经典 ODE 解映射。相应的 oracle 流映射记为 ${\boldsymbol{\Psi}}_{s\to t}$。经验流映射旨在近似 oracle 映射：

$$
\widetilde{{\boldsymbol{\Psi}}}_{s\to t}({\mathbf{x}}_s)
    \approx
    {\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s).
$$

由于[式 (9.1.1)](01-prologue.md#eq-solution-int-score)中的积分通常没有可计算的闭式形式，数值采样器使用递减网格离散逆时间区间

<span id="eq-time-step-def"></span>

$$
\begin{aligned}
T=t_0>t_1>\cdots>t_M=0,
\end{aligned}
\tag{9.1.2}
$$

从

$$
\tilde{{\mathbf{x}}}_{t_0}={\mathbf{x}}_T\sim p_{\mathrm{prior}},
$$

出发，产生序列

$$
\tilde{{\mathbf{x}}}_{t_0},\tilde{{\mathbf{x}}}_{t_1},\ldots,\tilde{{\mathbf{x}}}_{t_M},
$$

其中 $\tilde{{\mathbf{x}}}_{t_i}$ 近似 $\widetilde{\boldsymbol{\Psi}}_{T\to t_i}({\mathbf{x}}_T)$。最终迭代值 $\tilde{{\mathbf{x}}}_{t_M}$ 用作生成样本。

因此，本章中的每个免训练求解器都在回答同一个问题：

<span id="question-9-1-1"></span>

**问题 9.1.1**

如何用尽可能少的模型计算，准确近似 PF-ODE 积分？

**求解器设计中的三个独立选择。** 在引入[式 (9.1.2)](01-prologue.md#eq-time-step-def)中的递减时间网格后，有必要区分扩散采样中经常交织在一起的三个选择。

1.  **预测参数化。** 预训练扩散模型提供什么量：score ${\mathbf{s}}_{\boldsymbol{\phi}^\times}$、噪声 ${\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}$、干净数据 ${\mathbf{x}}_{\boldsymbol{\phi}^\times}$，还是速度 ${\mathbf{v}}_{\boldsymbol{\phi}^\times}$？

2.  **时间参数化。** 使用什么变量参数化同一连续轨迹上的进展？原始扩散时间 $t$ 是一种选择；后文会在需要时介绍其他重参数化。

3.  **数值近似。** 求解器如何在有限区间内近似未知的模型相关量？它可以只使用当前计算结果，在当前步内额外计算，或复用前面步骤的计算结果。

这一区分贯穿本章。前两个选择决定呈现给求解器的数值形式，第三个选择决定求解器如何使用可获得的模型计算结果。

**求解器如何使用模型计算结果？** 整理数值求解器的一种有用方式，是询问下一次更新的信息来自哪里。从当前状态出发，求解器可以在*当前步内*获取额外信息，也可以复用*先前步骤*的信息。

*单步方法*只使用当前状态及同一时间区间内产生的计算结果来构造下一状态。Euler 方法是最简单的例子，只使用当前速度。更高阶 Runge–Kutta（RK）方法则可能在完成该步之前，在一个或多个中间状态计算速度。因此，“单步”不表示“只计算一次函数”，而表示不需要先前已完成步骤的信息。

*多步方法*复用先前已完成步骤的信息。存储的模型计算结果提供了动力学最近如何变化的信息，使当前步无需多次新计算也能进行高阶更新。Adams–Bashforth（AB）族是经典代表。

另一种独立思想是改变跨时间更新的调度方式。可以先初始化多个时刻的状态，再反复校正，直到它们与 ODE 一致，而不必严格顺序计算所有时间点。这类不动点迭代允许在不同时间点并行计算模型。

Euler、RK、AB 和 Picard 迭代等经典代表将在[第 A.1.5 节](../../appendices/a-differential-equations/01-ordinary-differential-equations.md#app-de-solver-examples)回顾；[式 (A.1.6)](../../appendices/a-differential-equations/01-ordinary-differential-equations.md#eq-classic-solvers)总结了它们的结构分类。

**计算代价与实际表现。** 一个数值时间步不一定对应一次神经网络计算。Euler 每步计算一次，而高阶 RK 方法可能需要多次。多步方法复用存储的计算结果，初始化后通常每步只需一次新计算即可前进。

因此，我们用*函数计算次数*（number of function evaluations，NFE）度量采样代价。如果一种方法每步大约使用 $m$ 次模型计算，共进行 $M$ 步，则

$$
\mathrm{NFE}\approx mM.
$$

例如，无分类器引导（见[第 8.3 节](../08-guidance/03-classifier-free-guidance.md#sec-cfg)）同时需要条件与无条件模型预测，因此每个采样步的神经网络计算代价约翻倍。

这一经典分类说明了*数值信息来自哪里*以及需要多少次模型计算。扩散 PF-ODE 还提供了额外机会：向量场的不同部分并非同样未知。根据预测参数化，部分动力学由噪声调度解析确定，只有剩余量需要神经网络计算。接下来考察如何通过显露这一区分来改变 PF-ODE 的数值形式，并启发扩散专用积分器。

## 9.1.2 扩散专用求解器的设计 {#subsec-common-design-solvers}

[第 A.1.4 节](../../appendices/a-differential-equations/01-ordinary-differential-equations.md#app-de-solver)回顾的经典数值方法通过计算向量场处理一般 ODE。扩散 PF-ODE 具有额外结构：噪声调度解析已知，而扩散模型预测由神经网络提供。

这带来了有用的机会。求解器可以解析保留已知的调度相关项，只近似模型相关预测，无需以同样方式数值近似动力学的所有部分。这一结构显露得多清楚，取决于预测参数化。

**I. score 以外的参数化。** 尽管 PF-ODE 可以写成 score 形式，现代实现通常使用噪声预测、干净数据预测或速度预测。这些参数化在代数上相互联系，但呈现的数值结构不同。

许多求解器避免直接预测 score，还有一个重要的稳定性原因。当 $t\to 0$ 时，真实 score $\nabla_{{\mathbf{x}}}\log p_t(\cdot)$ 可能变化极快，尤其在数据分布集中于低维流形附近时（[Kim et al., 2022](../../references.md#ref-kim2022soft)）。这使直接近似 score 的神经网络难以在小噪声水平附近保持准确[^1]。

因此，广泛使用的替代方案是预测噪声 ${\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}$，或干净数据、速度等相关目标。在噪声预测形式下，score 近似为

$$
{\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}},t)
    =
    -\frac{1}{\sigma_t}\,
    {\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}},t).
$$

这些预测参数化在代数上相互联系，但神经网络提供的量只是求解器设计的一个组成部分。下面考察同一 PF-ODE 在不同参数化下的数值表现形式。

**II. 一个 PF-ODE，两种数值形式。** 选定预测参数化后，同一 PF-ODE 可以以两种特别有用的形式提供给数值求解器。

**直接速度形式。** 最直接的表示是

<span id="eq-direct-pfode-form"></span>

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t}
=
{\mathbf{v}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}(t),t).
\tag{9.1.3}
$$

这里，在 ${\mathbf{v}}$ 预测下，模型直接提供完整的经验 PF-ODE 速度 ${\mathbf{v}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}(t),t)$。因此，数值积分方法将整个右端项作为整体离散化。

**因子分解的半线性形式。** score、噪声和干净数据预测自然地显露了额外的调度相关结构。在每种情况下，PF-ODE 都可写为一般的*半线性*形式

<span id="eq-general-semilinear-pfode"></span>

$$
\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t}
=
\underbrace{
a(t){\mathbf{x}}(t)
}_{\substack{\text{known}\\\text{linear part}}}
+
\underbrace{
b(t)
}_{\substack{\text{known}\\\text{coefficient}}}
\underbrace{
{\mathbf{N}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}(t),t)
}_{\substack{\text{neural}\\\text{evaluation}}},
\quad
{\mathbf{N}}_{\boldsymbol{\phi}^\times}
\in
\left\{
{\mathbf{s}}_{\boldsymbol{\phi}^\times},
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times},
{\mathbf{x}}_{\boldsymbol{\phi}^\times}
\right\}.
\tag{9.1.4}
$$

这里，$a(t)$ 和 $b(t)$ 是由噪声调度确定的已知函数，${\mathbf{N}}_{\boldsymbol{\phi}^\times}$ 表示神经网络预测[^2]。*半线性*指的是这一分解：第一项关于状态 ${\mathbf{x}}(t)$ 线性，而神经网络项可能非线性依赖 ${\mathbf{x}}(t)$。这一分解并不唯一，但在数值上特别有用，因为已知线性部分通常可以解析处理，只留下神经网络贡献需要数值近似。

对于三种标准半线性参数化[^3]，相应的 $a(t)$、$b(t)$ 和 ${\mathbf{N}}_{\boldsymbol{\phi}^\times}$ 选择列于[表 9.1](01-prologue.md#tb-pf-ode-parameterizations)。例如，在噪声预测下，

<span id="eq-noise-pf-ode"></span>

$$
\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t}
=
\underbrace{
f(t){\mathbf{x}}(t)
}_{\text{linear part}}
+
\underbrace{
\frac{1}{2}\frac{g^2(t)}{\sigma_t}
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}(t),t)
}_{\text{nonlinear neural part}}.
\tag{9.1.5}
$$

神经网络预测前的系数完全由噪声调度确定；第二项中唯一学习到的量是 ${\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}(t),t)$。

<span id="tb-pf-ode-parameterizations"></span>

**表 9.1：score、噪声和干净数据预测下的因子分解 PF-ODE 形式。**

<table markdown="1" style="display: table; overflow: visible;">
<tr markdown="1">
<th markdown="block">

**预测**

</th>
<th markdown="block">

$\boldsymbol{a(t)}$

</th>
<th markdown="block">

$\boldsymbol{b(t)}$

</th>
<th markdown="block">

$\boldsymbol{{\mathbf{N}}_{\boldsymbol{\phi}^\times}}$

</th>
</tr>
<tr markdown="1">
<td markdown="block">

${\mathbf{s}}$ 预测

</td>
<td markdown="block">

$f(t)$

</td>
<td markdown="block">

$-\frac{1}{2}g^2(t)$

</td>
<td markdown="block">

${\mathbf{s}}_{\boldsymbol{\phi}^\times}$

</td>
</tr>
<tr markdown="1">
<td markdown="block">

${\boldsymbol{\epsilon}}$ 预测

</td>
<td markdown="block">

$f(t)$

</td>
<td markdown="block">

$\displaystyle \tfrac{1}{2}\tfrac{g^2(t)}{\sigma_t}$

</td>
<td markdown="block">

${\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}$

</td>
</tr>
<tr markdown="1">
<td markdown="block">

${\mathbf{x}}$ 预测

</td>
<td markdown="block">

$\displaystyle \tfrac{\sigma_t'}{\sigma_t}$

</td>
<td markdown="block">

$\displaystyle \alpha_t \left( \tfrac{\alpha_t'}{\alpha_t} - \tfrac{\sigma_t'}{\sigma_t} \right)$

</td>
<td markdown="block">

${\mathbf{x}}_{\boldsymbol{\phi}^\times}$

</td>
</tr>
</table>

在 oracle 层面，score、噪声、干净数据与速度预测是同一 PF-ODE 的完全等价表示。对于经验模型，当不同预测参数化由同一模型输出经精确代数转换得到时，这一等价性仍成立（见[式 (6.3.1)](../06-unified-perspective/03-equivalent-parameterizations.md#eq-predictions-equivalence)）。此时，同一经验 PF-ODE 可通过 score、噪声、干净数据或速度预测表达。所得公式看起来不同，是因为它们呈现了不同的数值结构，但描述的是相同的学习连续动力学。然而，如果针对不同预测目标独立训练不同网络，它们的经验 PF-ODE 不一定完全一致。这时，采样器之间的差异可能同时反映模型近似与数值离散化。

**III. 半线性 PF-ODE 的指数积分器。** [式 (9.1.4)](01-prologue.md#eq-general-semilinear-pfode)中的因子分解半线性形式，将解析已知的线性动力学与模型相关预测分开。对于 $s>t$，定义积分因子

$$
\mathcal{E}(s\shortrightarrow t)
:=
\exp\left(
\int_s^t a(u)\mathop{}\!\mathrm{d} u
\right).
$$

常数变易公式于是给出精确的经验 PF-ODE 流

<span id="eq-factored-variation-of-constants"></span>

$$
\widetilde{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
=
\mathcal{E}(s\shortrightarrow t){\mathbf{x}}_s
+
\int_s^t
\mathcal{E}(\tau\shortrightarrow t)
b(\tau)
{\mathbf{N}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)
\mathop{}\!\mathrm{d}\tau .
\tag{9.1.6}
$$

一般推导见[第 A.1.3 节](../../appendices/a-differential-equations/01-ordinary-differential-equations.md#app-exp-int-factor)。

例如，在噪声预测下，$a(t)=f(t)$、$b(t)=\tfrac{1}{2}g^2(t)/\sigma_t$ 且 ${\mathbf{N}}_{\boldsymbol{\phi}^\times}={\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}$，因此

<span id="eq-variation-of-constants"></span>

$$
\widetilde{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
=
\underbrace{
\mathcal{E}(s\shortrightarrow t){\mathbf{x}}_s
}_{\text{exact linear evolution}}
+
\frac{1}{2}
\int_s^t
\frac{g^2(\tau)}{\sigma_\tau}
\mathcal{E}(\tau\shortrightarrow t)
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)
\mathop{}\!\mathrm{d}\tau,
\tag{9.1.7}
$$

其中 $\mathcal{E}(s\shortrightarrow t) = \exp\!\left(\int_s^t f(u)\mathop{}\!\mathrm{d} u\right)$。

数值上的机会由此清晰可见。积分因子与系数 $b(\tau)$ 由扩散调度确定；积分中唯一未知的量，是沿轨迹计算的神经网络预测。显式保留这些已知因子，就能使一阶近似只作用于神经网络预测，而不把系数加权乘积 $b(\tau){\mathbf{N}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)$ 吸收到单个未知残差中。

**因子分解指数积分器形式为何有帮助。** 为理解这种分离的好处，将其与普通 Euler 方法比较：从精确经验 PF-ODE 状态 ${\mathbf{x}}_s$（时刻 $s$）向后走一步到 $t<s$。定义 $\Delta s:=s-t>0$ 和 ${\mathbf{N}}_s:={\mathbf{N}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_s,s)$。

普通 Euler 将完整 PF-ODE 速度视为局部常数，即 $a(\tau){\mathbf{x}}_\tau+ b(\tau){\mathbf{N}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau) \approx a(s){\mathbf{x}}_s+b(s){\mathbf{N}}_s$，其中 $\tau\in[t,s]$，得到

<span id="eq-euler-step"></span>

$$
\begin{aligned}
\tilde{\mathbf{x}}_t^{\mathrm{Euler}}
=
{\mathbf{x}}_s
-
\Delta s
\left[
a(s){\mathbf{x}}_s+b(s){\mathbf{N}}_s
\right].
\end{aligned}
\tag{9.1.8}
$$

也可以只固定神经网络预测，即 ${\mathbf{N}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)\approx{\mathbf{N}}_s$，同时保留[式 (9.1.6)](01-prologue.md#eq-factored-variation-of-constants)中已知的调度依赖。这得到*一阶因子分解指数积分更新*[^4]

<span id="eq-factored-ei-step"></span>

$$
\begin{aligned}
\tilde{\mathbf{x}}_t^{\mathrm{EI}}
=
\mathcal E(s\shortrightarrow t){\mathbf{x}}_s
+
\left[
\int_s^t
\mathcal E(\tau\shortrightarrow t)b(\tau)\mathop{}\!\mathrm{d}\tau
\right]{\mathbf{N}}_s.
\end{aligned}
\tag{9.1.9}
$$

两种方法每步都使用一次神经网络计算。普通 Euler 冻结完整 PF-ODE 速度，因子分解更新则保留已知调度依赖，只冻结神经网络预测。这一分解很关键：如果将 $b(\tau){\mathbf{N}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)$ 视为单个残差并固定，就得到通常的指数 Euler。因此，因子分解更新在积分内多保留了一个已知量 $b(\tau)$；当 $b(\tau)$ 在该步内为常数时，两种更新一致。

一个简单的冻结预测例子可以具体说明这一区别。假设一步之内 $a(t)\equiv a$、$b(t)\equiv b$ 且 ${\mathbf{N}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)\equiv{\mathbf{N}}_s$。对于 $a\neq0$，

$$
\begin{aligned}
\tilde{\mathbf{x}}_t^{\mathrm{EI}}
&=
e^{-a\Delta s}{\mathbf{x}}_s
+
\frac{b}{a}
\left(
e^{-a\Delta s}-1
\right){\mathbf{N}}_s,
\\
\tilde{\mathbf{x}}_t^{\mathrm{Euler}}
&=
(1-a\Delta s){\mathbf{x}}_s
-
b\Delta s\,{\mathbf{N}}_s.
\end{aligned}
$$

展开因子分解更新，得到

$$
\begin{aligned}
\tilde{\mathbf{x}}_t^{\mathrm{EI}}
=
\tilde{\mathbf{x}}_t^{\mathrm{Euler}}
+
\frac{(\Delta s)^2}{2}
\left[
a^2{\mathbf{x}}_s
+
ab\,{\mathbf{N}}_s
\right]
+
\mathcal O((\Delta s)^3).
\end{aligned}
$$

因此，在步长很小时，两种更新在一阶上相符，而因子分解更新保留了已知动力学的高阶影响。当步长较大时，这种差别可能很重要。例如，当 $a>0$ 时，精确乘子 $e^{-a\Delta s}$ 对每个 $\Delta s>0$ 都保持为正且具有收缩性，而 Euler 近似 $1-a\Delta s$ 在 $a\Delta s>1$ 时变为负数，在 $a\Delta s>2$ 时其绝对值大于一。因此，普通 Euler 可能扭曲那些因子分解形式已经解析处理的动力学。

在完整 PF-ODE 中，神经网络预测一般会沿轨迹变化。用精确常数变易公式减去[式 (9.1.9)](01-prologue.md#eq-factored-ei-step)，得到

<span id="eq-ei-local-defect"></span>

$$
\begin{aligned}
\widetilde{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
-
\tilde{\mathbf{x}}_t^{\mathrm{EI}}
=
\int_s^t
\mathcal E(\tau\shortrightarrow t)b(\tau)
\Big[
{\mathbf{N}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)
-
{\mathbf{N}}_s
\Big]\mathop{}\!\mathrm{d}\tau.
\end{aligned}
\tag{9.1.10}
$$

因此，一旦解析处理已知的调度相关动力学，剩余离散化误差就由神经网络预测沿轨迹的变化控制。这种分离是若干扩散专用求解器将以不同方式利用的基本数值原理。

## 9.1.3 从数值思想到扩散采样器的路线 {#roadmap}

本章余下内容展示这些数值思想如何体现在现代扩散采样器中。[第 9.2 节](02-ddim.md#sec-revisit-ddim)首先介绍 DDIM，它根据当前模型预测构造一阶更新。[第 9.3 节](03-deis.md#sec-exponential)研究 DEIS，它使用先前步骤的预测构造高阶近似。[第 9.4 节](04-dpm-solver.md#sec-dpm)中的 DPM-Solver 则通过当前步内额外的模型计算构造高阶更新。[第 9.5 节](05-dpm-solver-plus-plus.md#sec-dpm)中的 DPM-Solver`++` 基于干净数据预测发展了两个对应变体：单步变体 DPM-Solver`++`(2S) 在当前步内增加计算，多步变体 DPM-Solver`++`(2M) 则复用先前步骤的预测。最后，[第 9.6 节](06-paradigms.md#sec-picard)介绍 ParaDiGMS，它通过不动点迭代同时更新多个时间点，以实现跨时间并行计算。

下表将这些扩散采样器与[式 (A.1.6)](../../appendices/a-differential-equations/01-ordinary-differential-equations.md#eq-classic-solvers)回顾的经典数值思想联系起来。

<table markdown="1" style="display: table; overflow: visible;">
<tr markdown="1">
<th markdown="block">

**经典思想**

</th>
<th markdown="block">

**更新的构造方式**

</th>
<th markdown="block">

**扩散方法示例**

</th>
<th markdown="block">

**所在位置**

</th>
</tr>
<tr markdown="1">
<td markdown="block">

Euler 型一阶更新

</td>
<td markdown="block">

使用该步起点的模型预测推进至下一状态。

</td>
<td markdown="block">

DDIM; DPM-Solver-1

</td>
<td markdown="block">

[第 9.2 节](02-ddim.md#sec-revisit-ddim) 和[第 9.4 节](04-dpm-solver.md#sec-dpm)

</td>
</tr>
<tr markdown="1">
<td markdown="block">

RK 型分阶段更新

</td>
<td markdown="block">

在完成当前步之前，于额外的中间状态计算模型。

</td>
<td markdown="block">

高阶 DPM-Solver; DPM-Solver`++`(2S)

</td>
<td markdown="block">

[第 9.4 节](04-dpm-solver.md#sec-dpm) 和[第 9.5 节](05-dpm-solver-plus-plus.md#sec-dpm)

</td>
</tr>
<tr markdown="1">
<td markdown="block">

AB 型多步更新

</td>
<td markdown="block">

复用先前步骤的模型预测来构造下一次更新。

</td>
<td markdown="block">

DEIS; DPM-Solver`++`(2M)

</td>
<td markdown="block">

[第 9.3 节](03-deis.md#sec-exponential) 和[第 9.5 节](05-dpm-solver-plus-plus.md#sec-dpm)

</td>
</tr>
<tr markdown="1">
<td markdown="block">

Picard 型不动点迭代

</td>
<td markdown="block">

同时更新多个时间点，反复改进其状态。

</td>
<td markdown="block">

ParaDiGMS

</td>
<td markdown="block">

[第 9.6 节](06-paradigms.md#sec-picard)

</td>
</tr>
</table>

这些联系为下文方法提供了整体数值路线。扩散求解器还受益于 PF-ODE 特有的结构，包括不同预测参数化、解析已知的调度相关系数，以及其他时间参数化。

为使介绍系统化，各节遵循相同的整体结构：先推导扩散求解器，解释其利用的数值思想；再将它与相应经典求解器原理联系；最后说明它与本章其他扩散求解器的关系。这种组织方式同时凸显共同数值基础，以及使各方法产生差异的扩散特有要素。

[^1]: 为了解缩放关系，回顾[式 (6.3.1)](../06-unified-perspective/03-equivalent-parameterizations.md#eq-predictions-equivalence)中的 oracle 关系：

    $$
    {\boldsymbol{\epsilon}}^*({\mathbf{x}}_t,t)
        =
        -\sigma_t\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}}_t),
    $$

    其中，${\boldsymbol{\epsilon}}^*({\mathbf{x}}_t,t) = \mathbb{E}[{\boldsymbol{\epsilon}}|{\mathbf{x}}_t]$ 是 oracle 噪声预测器，扰动核为 ${\mathbf{x}}_t|{\mathbf{x}}_0 \sim \mathcal{N}({\mathbf{x}}_t; \alpha_t{\mathbf{x}}_0,\sigma_t^2{\mathbf{I}})$。根据 $L^2$ 中条件期望的正交性质，

    $$
    \mathbb{E}\|{\boldsymbol{\epsilon}}\|_2^2
        =
        \mathbb{E}\|{\boldsymbol{\epsilon}}^*\|_2^2
        +
        \mathbb{E}\|{\boldsymbol{\epsilon}}-{\boldsymbol{\epsilon}}^*\|_2^2,
    $$

    因此

    $$
    \mathbb{E}\|{\boldsymbol{\epsilon}}^*\|_2^2
        \le
        \mathbb{E}\|{\boldsymbol{\epsilon}}\|_2^2
        =
        D,
    $$

    其中，$D$ 是数据维数。因此，oracle 噪声预测器的期望平方范数具有一致上界。相比之下，oracle score

    $$
    {\mathbf{s}}^*({\mathbf{x}}_t,t)
        :=
        \nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}}_t)
    $$

    满足

    $$
    \mathbb{E}\|{\mathbf{s}}^*({\mathbf{x}}_t,t)\|_2^2
        =
        \sigma_t^{-2}\,
        \mathbb{E}\|{\boldsymbol{\epsilon}}^*({\mathbf{x}}_t,t)\|_2^2
        \le
        \frac{D}{\sigma_t^2}.
    $$

    因此，随着 $\sigma_t$ 变小，直接 score 预测的尺度可能越来越不良，因为噪声预测通过因子 $1/\sigma_t$ 转换为 score。这种缩放会使直接 score 参数化在小噪声水平附近的数值表现较差。

[^2]: 在标准半线性记号中，可以将 $b(t){\mathbf{N}}_{\boldsymbol{\phi}^\times}$ 吸收到单个非线性项中。这里保留两个独立因子，是因为 $b(t)$ 由噪声调度解析给出，而 ${\mathbf{N}}_{\boldsymbol{\phi}^\times}$ 需要神经网络计算。这使求解器可以在积分内近似神经网络预测，同时保留并解析积分已知的调度相关系数。

[^3]: 后续各节会在 SDE 系数 $(f(t),g(t))$ 与 Gaussian 扰动核 ${\mathbf{x}}_t | {\mathbf{x}}_0 \sim \mathcal{N} \bigl( {\mathbf{x}}_t; \alpha_t{\mathbf{x}}_0, \sigma_t^2{\mathbf{I}} \bigr)$ 之间切换。二者关系为

    $$
    f(t)
        =
        \frac{\alpha_t'}{\alpha_t},
        \qquad
        g^2(t)
        =
        \frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}\bigl(\sigma_t^2\bigr)
        -
        2\frac{\alpha_t'}{\alpha_t}\sigma_t^2
        =
        2\sigma_t\sigma_t'
        -
        2\frac{\alpha_t'}{\alpha_t}\sigma_t^2.
    $$

[^4]: [式 (9.1.9)](01-prologue.md#eq-factored-ei-step)中的更新并非原始状态与时间变量下的普通 Euler。对状态作积分因子变换，并在所得时间坐标单调的区间内作相应时间重参数化后，可以将其写为普通显式 Euler 步。DDIM 一节会具体说明这一联系在扩散采样中的体现。
