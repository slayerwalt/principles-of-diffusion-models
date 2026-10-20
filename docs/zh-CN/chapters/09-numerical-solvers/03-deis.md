# 9.3 DEIS {#sec-exponential}

在因子分解指数积分公式（[式 (9.1.7)](01-prologue.md#eq-variation-of-constants)）中，

$$
\int_s^t
\frac{g^2(\tau)}{2\sigma_\tau}\,
\mathcal E(\tau\shortrightarrow t)\,
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)
\mathop{}\!\mathrm{d}\tau,
$$

所有调度相关因子都已知；沿轨迹唯一未知的量是神经网络预测 ${\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)$。DDIM 将这一预测固定为该步起点的值，作出最简单的一阶近似，

$$
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)
\approx
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_s,s),
\qquad
\tau\in[t,s].
$$

当神经网络预测沿轨迹明显变化时，这一近似在较大步长下可能不准确。

DEIS（[Zhang and Chen, 2022](../../references.md#ref-zhang2022fast)）改进了这一思想，而不在当前步内引入额外模型计算。它受 Adams–Bashforth 等经典多步方法启发（见[第 A.1.5 节](../../appendices/a-differential-equations/01-ordinary-differential-equations.md#app-de-solver-examples)），复用先前已完成步骤的模型预测作为*锚点*，拟合多项式，将近期变化外推到下一步。随后用多项式替代积分内未知的神经网络预测，同时解析保留已知调度相关权重 $\tfrac{g^2(\tau)}{2\sigma_\tau} \mathcal E(\tau\shortrightarrow t)$。因此，DEIS 将 DDIM 的常数近似扩展为高阶多步多项式近似。

从经典数值视角看，DEIS 是因子分解指数积分框架中的 Adams–Bashforth 型多步方法：先前的神经网络计算提供高阶信息，已知扩散动力学则保留在加权积分中。

我们先在[第 9.3.1 节](03-deis.md#subsec-poly-extrapolation)回顾用于构造这一近似的多项式外推，再在[第 9.3.2 节](03-deis.md#subsec-deis-lagrange)推导所得 DEIS 更新。最后，[第 9.3.3 节](03-deis.md#sec-ddim-deis)说明常数即零次多项式情形会恢复 DDIM。

## 9.3.1 背景：多项式外推 {#subsec-poly-extrapolation}

推导 DEIS 之前，先简要回顾它用于复用先前模型计算的多项式构造。思想很简单：先拟合一个通过若干已知值的多项式，再在最近锚点之外计算该多项式，预测这一量可能如何继续变化。因此，该多项式在已知锚点上是*插值多项式*，在下一个求解器步骤上则充当*外推多项式*。

**从锚点到外推多项式。** 假设某个时变量在 $n{+}1$ 个不同的时刻已知，

$$
(\tau_0,{\mathbf{Y}}_0),\;
(\tau_1,{\mathbf{Y}}_1),\;
\ldots,\;
(\tau_n,{\mathbf{Y}}_n),
\qquad
\tau_0<\tau_1<\cdots<\tau_n,
$$

其中，每个 ${\mathbf{Y}}_j$ 都可以是向量值。我们希望构造最简单的光滑曲线，精确匹配这些观测值，并可延伸到最近锚点之外。通过全部锚点、次数至多为 $n$ 的唯一多项式是 *Lagrange 插值多项式*

$$
P_n(\tau)
=
\sum_{j=0}^{n}
\ell_j(\tau)\,{\mathbf{Y}}_j,
$$

其基函数为

$$
\ell_j(\tau)
:=
\prod_{\substack{k=0\\k\neq j}}^{n}
\frac{\tau-\tau_k}{\tau_j-\tau_k}.
$$

$\ell_j$ 的构造很简单：每个因子都使它在某个其他锚点处为零，分母则将其值归一化为 $1$，对应位置为 $\tau_j$。因此，

$$
\ell_j(\tau_k)=\delta_{jk},
$$

从而

$$
P_n(\tau_j)={\mathbf{Y}}_j,
\qquad
j=0,\ldots,n.
$$

此外，

$$
\sum_{j=0}^{n}\ell_j(\tau)=1.
$$

因此，$P_n(\tau)$ 是观测锚点随时间变化的线性组合：在每个锚点时刻精确重现对应值，在锚点之间或之外，则通过组合锚点描述其变化所暗示的局部趋势。DEIS 将在最近锚点之外使用同一曲线，使插值多项式成为下一步的*外推多项式*。

最低次的几个情形可以说明这一构造：

**示例。**

**$n=0$（常数）。** 只有一个锚点时，多项式只是保持最近值不变：

$$
P_0(\tau)={\mathbf{Y}}_0.
$$

**$n=1$（线性）。** 有两个锚点时，多项式就是连接二者的直线：

$$
P_1(\tau)
=
\frac{\tau-\tau_1}{\tau_0-\tau_1}{\mathbf{Y}}_0
+
\frac{\tau-\tau_0}{\tau_1-\tau_0}{\mathbf{Y}}_1.
$$

**$n=2$（二次）。** 有三个锚点时，插值多项式为

$$
P_2(\tau)
=
\ell_0(\tau){\mathbf{Y}}_0
+
\ell_1(\tau){\mathbf{Y}}_1
+
\ell_2(\tau){\mathbf{Y}}_2,
$$

它还可以捕捉观测值的局部曲率。

DEIS 的锚点是先前已完成采样步骤中的神经网络预测。多项式精确匹配这些存储的预测，然后在下一个逆时间区间内，从最近锚点向外外推。

## 9.3.2 DEIS：PF-ODE 积分的 Lagrange 多项式近似 {#subsec-deis-lagrange}

令 $n \ge 0$ 为选定的多项式次数。在第 $i$ 步，DEIS 近似未知映射 $\tau \mapsto {\boldsymbol{\epsilon}}_{{\boldsymbol{\phi}}^\times}({\mathbf{x}}_\tau,\tau)$ 在下一个逆时间区间 $[t_i,t_{i-1}]$ 上的变化，所用的是由过去模型输出构造的 $n$ 次 Lagrange 多项式。随后将此多项式代入指数积分更新（[式 (9.1.7)](01-prologue.md#eq-variation-of-constants)），得到 $\tilde{{\mathbf{x}}}_{t_i}$。直观上，DEIS 利用神经网络预测的近期历史来捕捉其随时间的变化，而不像 DDIM 那样将其保持为常数。

<span id="fig-deis-lagrange"></span>

![DEIS 用二次多项式外推三个历史锚点处的模型输出](../../../en/assets/chapter-09/deis-multistep.png)

**图 9.2：** **DEIS 作为多步方法的示意图。** 给定 $t_2,t_3,t_4$ 处的三个历史锚点，DEIS 构造通过模型输出的二次曲线，并在从 $t_4$ 到 $t_5$ 的步骤内进行外推。随后对外推多项式乘以已知扩散相关权重进行积分，捕捉 DDIM 常数近似忽略的神经网络预测变化。

来源：作者绘制。

$n$ 次更新需要 $n{+}1$ 个锚点。锚点齐备时（*历史信息充分*，$i \ge n{+}1$），使用完整的 $n$ 次格式。在初始步骤中（*历史信息不足*，$i \le n$），使用可行的最高次数 $i{-}1$，并随着更多锚点出现逐渐提高次数。下面分别处理这两种情况。

**情况 I：$i = n+1,\dots,M$（历史信息充分）。** 历史充分时，DEIS 复用最近 $n{+}1$ 次模型计算作为锚点，

$$
(\tau_j,{\mathbf{Y}}_j) := \bigl(t_{i-1-j}, \boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}(\tilde{{\mathbf{x}}}_{t_{i-1-j}},t_{i-1-j})\bigr),
\quad j=0,\dots,n.
$$

将 $\tau\mapsto\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)$ 视为沿轨迹的光滑函数，构造 $n$ 次 Lagrange 多项式

$$
P_n(\tau)
=
\sum_{j=0}^{n}
\underbrace{\Bigl[\prod_{\substack{k=0\\k\neq j}}^{n}
\frac{\tau - t_{i-1-k}}{\,t_{i-1-j} - t_{i-1-k}\,}\Bigr]}_{=:~\ell_j^{(i)}(\tau)}
\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}\bigl(\tilde{\mathbf x}_{t_{i-1-j}}, t_{i-1-j}\bigr)
$$

根据构造，它在每个锚点满足 $P_n(\tau_j)={\mathbf{Y}}_j$：

$$
P_n\bigl(\tau_j\bigr)
= {\mathbf{Y}}_j =
\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}\bigl(\tilde{\mathbf x}_{t_{i-1-j}}, t_{i-1-j}\bigr),
\quad
 j=0,\ldots,n.
$$

每个 $\ell_j^{(i)}$ 都满足

$$
\ell_j^{(i)}\bigl(t_{i-1-m}\bigr)
=
\begin{cases}
1,&m=j,\\
0,&m\neq j.
\end{cases}
$$

然后将多项式从最近锚点 $t_{i-1}$ 向外外推到新步骤内：

$$
\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau) \approx
P_n(\tau)=\sum_{j=0}^{n}\ell_j^{(i)}(\tau)
\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}\bigl(\tilde{{\mathbf{x}}}_{t_{i-1-j}}, t_{i-1-j}\bigr),
\quad \tau \in [t_i,t_{i-1}].
$$

接着在指数积分公式（[式 (9.1.7)](01-prologue.md#eq-variation-of-constants)）中，用 $P_n(\tau)$ 替换 ${\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)$：

$$
\begin{aligned}
\int_{t_{i-1}}^{t_i}
\frac{g^2(\tau)}{2\sigma_\tau}
\mathcal{E}(\tau\to t_i)
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)\mathop{}\!\mathrm{d}\tau
&\approx
\sum_{j=0}^{n}
\underbrace{
\int_{t_{i-1}}^{t_i}
\frac{g^2(\tau)}{2\sigma_\tau}
\mathcal{E}(\tau\to t_i)
\ell_j^{(i)}(\tau)\mathop{}\!\mathrm{d}\tau
}_{=:~C_{i,j}}
\\
&\qquad\cdot
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}
(\tilde{\mathbf{x}}_{t_{i-1-j}},t_{i-1-j}).
\end{aligned}
$$

系数

$$
C_{i,j}
:=
\frac{1}{2}
\int_{t_{i-1}}^{t_i}
\frac{g^2(\tau)}{\sigma_\tau}
\mathcal{E}(\tau\to t_i)
\ell_j^{(i)}(\tau)\mathop{}\!\mathrm{d}\tau
$$

只依赖噪声调度及选定时间网格，因此可以预先计算。对于具有合适原函数的调度，这些系数有闭式形式；否则，可以通过高精度数值计算得到。

利用 $\mathcal{E}(t_{i-1}\to t_i)$ 精确积分线性部分，得到 *AB-DEIS-$n$* 更新

$$
\tilde{\mathbf{x}}_{t_i}
=
\mathcal{E}(t_{i-1}\to t_i)\tilde{\mathbf{x}}_{t_{i-1}}
+
\sum_{j=0}^{n}
C_{i,j}
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}
(\tilde{\mathbf{x}}_{t_{i-1-j}},t_{i-1-j}).
$$

对于下述局部截断结论，令 $h$ 表示特征局部步长，并假设相邻步长比保持有界。在经验轨迹上的模型预测满足标准光滑性假设、调度相关权重满足正则性条件时，历史充分的 $n$ 次构造具有形式阶数 $n+1$，单步局部误差为 $\mathcal O(h^{n+2})$。

**情况 II：$i=1,\dots,n$（历史信息不足）。** 初始步骤中，仅有 $i$ 次过去模型计算可用。因此，使用所有可用锚点，并将多项式次数设为 $i{-}1$：

$$
P_{i-1}(\tau)
=
\sum_{j=0}^{i-1}
\ell_j^{(i)}(\tau)
\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}\bigl(\tilde{\mathbf x}_{t_{i-1-j}},  t_{i-1-j}\bigr),
$$

其中，$\ell_j^{(i)}$ 是 $i{-}1$ 次 Lagrange 基，构建于时刻 $\{t_{i-1}, t_{i-2},\ldots,t_{0}\}$ 的节点上。它匹配全部可用锚点，并在 $i\ge n{+}1$ 时转入历史充分的公式。

这是多步方法标准的启动情形。第一步使用常数近似，第二步使用线性外推，第三步使用二次外推，依此类推，直到能够使用目标次数 $n$。

**示例：Lagrange 多项式的特例**

**次数 $0$（一个锚点）：**

$$
P_0(\tau) = \boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_{t_{i-1}}, t_{i-1}).
$$

它将神经网络预测固定为该步起点的值，与一阶 DDIM 近似完全相同。

**次数 $1$（两个锚点）：** Lagrange 多项式是通过两个预先指定锚点的线性映射。

$$
P_1(\tau)
=
\underbrace{\frac{\tau - t_{i-2}}{t_{i-1} - t_{i-2}}}_{\displaystyle \ell_{i-1}(\tau)}
\,\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_{t_{i-1}}, t_{i-1})
 +
\underbrace{\frac{\tau - t_{i-1}}{t_{i-2} - t_{i-1}}}_{\displaystyle \ell_{i-2}(\tau)}
\,\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_{t_{i-2}}, t_{i-2}).
$$

这里，$\ell_{i-1}(\tau)$ 与 $\ell_{i-2}(\tau)$ 是 Lagrange 基权重。它们满足插值条件 $P_1(t_{i-1})=\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_{t_{i-1}}, t_{i-1})$ 和 $P_1(t_{i-2})=\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_{t_{i-2}}, t_{i-2})$，且 $\ell_{i-1}(\tau)+\ell_{i-2}(\tau)=1$。

**AB-DEIS-$n$ 更新汇总。** 结合历史充分与启动情形，得到 *AB-DEIS-$n$* 更新[^1]。这里，$n$ 是目标多项式次数，最多使用 $n{+}1$ 次历史计算：

$$
\begin{aligned}
\tilde{\mathbf{x}}_{t_i}
= \mathcal{E}(t_{i-1} \to t_i)\,\tilde{\mathbf{x}}_{t_{i-1}}
+ \sum_{j=0}^{\min\{n,\, i-1\}} C_{i,j}\,{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_{t_{i-1-j}},\,t_{i-1-j}),
\end{aligned}
$$

其系数为

$$
\begin{aligned}
C_{i,j}
:= \frac{1}{2} \int_{t_{i-1}}^{t_i}
\frac{g^2(\tau)}{\sigma_\tau}\,\mathcal{E}(\tau \to t_i)\,
\Bigg[\prod_{\substack{k=0\\k\neq j}}^{\min\{n,\, i-1\}}
\frac{\tau - t_{i-1-k}}{\,t_{i-1-j} - t_{i-1-k}\,}\Bigg]\,d\tau.
\end{aligned}
$$

当 $i\ge n{+}1$（历史充分）时，$\min\{n,\,i-1\}=n$，且在上述假设下，历史充分的 $n$ 次步骤具有单步局部截断误差 $\mathcal O(h^{n+2})$，对应 $n+1$ 阶方法。

启动期间（$i\le n$），次数为 $i-1$，所以单步局部截断误差为 $\mathcal O(h^{i+1})$，对应步骤的形式阶数为 $i$。因此，阶数从第一步的 $1$ 开始提高，并提高到完整阶数 $n+1$，此时已获得 $n+1$ 个锚点。与标准多步方法一样，较低阶的启动步骤本身不能建立全局 $(n+1)$ 阶保证；还需要稳定性及足够准确的起始值。

$n$ 过大可能使数值表现变差，因为高次外推会对锚点间距、存储模型预测的误差及稳定性问题越来越敏感。因此，实践中偏好相对较低的多项式次数。

下一小节将说明，特例 $n{=}0$ 恰好化为确定性 DDIM，即一阶因子分解指数积分更新。

## 9.3.3 DDIM = AB-DEIS-0 {#sec-ddim-deis}

在次数为 $n=0$ 时，与 DDIM 的联系立刻显现。这时 Lagrange 多项式为常数，因此 DEIS 在整个下一步中只使用最近一次模型预测。相应系数简化为

$$
C_{i0} = \frac{1}{2} \int_{t_{i-1}}^{t_{i}} \frac{g^2(\tau)}{\sigma_\tau} \mathcal{E}(\tau \shortrightarrow t_i)   \mathrm{d}\tau.
$$

代入更新公式，得到零次 AB-DEIS 格式：

<span id="eq-deis-0"></span>

$$
\begin{aligned}
\tilde{\mathbf{x}}_{t_{i}} &=  \mathcal{E}(t_{i-1} \shortrightarrow t_{i})\tilde{\mathbf{x}}_{t_{i-1}} + C_{i0} \boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_{t_{i-1}}, t_{i-1})\\
  &=e^{\int_{t_{i-1}}^{t_{i}}f(u)\mathop{}\!\mathrm{d} u} \tilde{\mathbf{x}}_{t_{i-1}} + \Big( \int_{t_{i-1}}^{t_{i}} \frac{g^2(\tau)}{2\sigma_\tau} e^{\int_{\tau}^{t_{i}}f(u)\mathop{}\!\mathrm{d} u}   \mathrm{d}\tau
\Big) \boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_{t_{i-1}}, t_{i-1}).
\end{aligned}
\tag{9.3.1}
$$

这恰好就是确定性 DDIM 使用的一阶因子分解近似：将神经网络预测固定在 $t_{i-1}$，而已知积分因子和调度相关系数仍保留在积分内，并在整个步骤上传播。因此，AB-DEIS-0 与 DDIM 不仅在一阶上等价，而且给出相同的有限步更新。

<span id="ddim-deis"></span>

**命题 9.3.1：DDIM = AB-DEIS-0**

[式 (9.3.1)](03-deis.md#eq-deis-0)与[式 (9.2.1)](02-ddim.md#eq-ddim-euler)中的 DDIM 更新完全相同。

这也澄清了 DEIS 在求解器体系中的作用。下标“$0$”指*多项式次数*，并非数值阶数：AB-DEIS-0 仍是一阶方法。更高次的 DEIS 保持对已知扩散动力学的相同解析处理，但用额外历史模型计算的多项式外推替代常数预测。因此，DDIM 是零次基础情形，DEIS 的多步构造由此自然扩展。

后面的[第 9.5.4 节](05-dpm-solver-plus-plus.md#subsec-dpm-solver--multi)将介绍另一条构造多步扩散求解器的途径，即 DPM-Solver`++`(2M)。它从对数信噪比时间中数据预测的局部 Taylor 展开出发，通过后向有限差分复用先前模型计算，估计所需导数。尽管表示与推导不同，两种方法都使用历史模型计算来捕捉神经网络预测的变化，而不增加步内阶段。[第 9.5.5 节](05-dpm-solver-plus-plus.md#subsec-dpm-deis)将再次讨论这一联系，并更明确地比较两种构造。

[^1]: “AB”指 Adams–Bashforth 多步原理，这里将其应用于指数积分表述（[Hochbruck and Ostermann, 2010](../../references.md#ref-hochbruck2010exponential)）。
