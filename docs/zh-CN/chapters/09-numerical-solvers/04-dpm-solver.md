# 9.4 DPM-Solver {#sec-dpm}

DPM-Solver（[Lu et al., 2022](../../references.md#ref-lu2022dpm)）利用前面讨论 DDIM 时出现的扩散特有结构，为经验 PF-ODE 构造高阶求解器。核心目标是以远少于一阶采样器的神经网络计算次数，获得准确样本。原始 DPM-Solver 关注噪声预测，构造了一阶、二阶和三阶单步求解器；后续扩展包括 DPM-Solver`++`（[Lu et al., 2022](../../references.md#ref-lu2022dpm2)）及 DPM-Solver-v3（[Zheng et al., 2023](../../references.md#ref-zheng2023dpm)），进一步发展了这一系列。

**DPM-Solver 的整体思想。** [第 9.2 节](02-ddim.md#sec-revisit-ddim)关于 DDIM 的讨论表明，在 ${\boldsymbol{\epsilon}}$ 预测下，解析保留已知扩散调度、同时在每步内固定神经网络预测，可以得到一阶更新，

$$
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)
\approx
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s).
$$

DPM-Solver（[Lu et al., 2022](../../references.md#ref-lu2022dpm)）自然延续这一思想：它不再将神经网络预测视为常数，而是对预测沿轨迹的变化构造高阶近似。与 DEIS 一样，它从 PF-ODE 的半线性 ${\boldsymbol{\epsilon}}$ 预测形式出发，

<span id="eq-noise-alpha-sigma"></span>

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}_t}{\mathop{}\!\mathrm{d} t}
=
\frac{\alpha_t'}{\alpha_t}{\mathbf{x}}_t
-
\sigma_t
\left(
\frac{\alpha_t'}{\alpha_t}
-
\frac{\sigma_t'}{\sigma_t}
\right)
{\boldsymbol{\epsilon}}_{{\boldsymbol{\phi}}^\times}({\mathbf{x}}_t,t),
\end{aligned}
\tag{9.4.1}
$$

并采用引言介绍的常数变易表示。其关键新增要素是沿用 VDM 的对数信噪比参数化（[Kingma et al., 2021](../../references.md#ref-kingma2021variational)），用*半对数信噪比*重新参数化时间，

<span id="eq-lambda"></span>

$$
\begin{aligned}
\lambda_t
:=
\log\frac{\alpha_t}{\sigma_t}.
\end{aligned}
\tag{9.4.2}
$$

这一坐标直接联系到 DDIM 的调度适配时钟 $\rho_t=\sigma_t/\alpha_t$（见[式 (9.2.4)](02-ddim.md#eq-ddim-coordinate-ode-t)附近的讨论），关系为 $\rho_t=e^{-\lambda_t}$，并将剩余模型相关项转化为特别方便的指数加权积分。随后可以解析积分神经网络预测在 $\lambda$ 中的局部 Taylor 近似，从而提供一条从一阶 DDIM 更新走向高阶扩散求解器的系统途径。

## 9.4.1 DPM-Solver 的思路：通过对数信噪比重参数化时间 {#subsec-dpm-method}

现在精确定义[式 (9.4.2)](04-dpm-solver.md#eq-lambda)中的对数信噪比重参数化，并推导相应 PF-ODE 表示。

**PF-ODE 中向对数信噪比的变量替换。** 假设 $\alpha_t>0$、$\sigma_t>0$，且[式 (9.4.2)](04-dpm-solver.md#eq-lambda)中的对数信噪比坐标 $\lambda_t$ 在所考虑区间上严格单调。那么它具有逆函数 $t_\lambda(\cdot)$，满足

$$
t=t_\lambda(\lambda(t)).
$$

对于满足 $\sigma_0=0$ 的调度，有 $\lambda_t\to+\infty$，对应极限为 $t\to0$，因此涉及端点 $t=0$ 的公式应通过相应极限理解，或在数值实现中使用一个很小的正终止时间。

随后在 $\lambda$ 坐标中表达状态、调度及神经网络预测。帽号（$\hat{\cdot}$）连同 $\lambda$ 表示对应量是在这一重参数化时间坐标下考察的。更精确地，定义

<span id="eq-change-of-var-x-epsilon"></span>

$$
\begin{aligned}
\hat{{\mathbf{x}}}_{\lambda}
&:=
{\mathbf{x}}_{t_\lambda(\lambda)},
\\
\alpha_\lambda
&:=
\alpha_{t_\lambda(\lambda)},
\qquad
\sigma_\lambda
:=
\sigma_{t_\lambda(\lambda)},
\\
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}},\lambda)
&:=
\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}
\bigl(
{\mathbf{x}},
t_\lambda(\lambda)
\bigr).
\end{aligned}
\tag{9.4.3}
$$

因此，沿经验 PF-ODE 轨迹，

$$
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda)
=
\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}
\bigl(
{\mathbf{x}}_{t_\lambda(\lambda)},
t_\lambda(\lambda)
\bigr).
$$

由于

$$
\frac{\mathop{}\!\mathrm{d}\lambda_t}{\mathop{}\!\mathrm{d} t}
=
\frac{\alpha_t'}{\alpha_t}
-
\frac{\sigma_t'}{\sigma_t},
$$

[式 (9.4.1)](04-dpm-solver.md#eq-noise-alpha-sigma)中的 PF-ODE 可写为

$$
\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}_t}{\mathop{}\!\mathrm{d} t}
=
\frac{\mathop{}\!\mathrm{d}\log\alpha_t}{\mathop{}\!\mathrm{d} t}{\mathbf{x}}_t
-
\sigma_t
\frac{\mathop{}\!\mathrm{d}\lambda_t}{\mathop{}\!\mathrm{d} t}
\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t,t).
$$

应用链式法则，

$$
\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}_t}{\mathop{}\!\mathrm{d} t}
=
\frac{\mathop{}\!\mathrm{d}\hat{\mathbf{x}}_\lambda}{\mathop{}\!\mathrm{d}\lambda}
\frac{\mathop{}\!\mathrm{d}\lambda_t}{\mathop{}\!\mathrm{d} t},
\qquad
\frac{\mathop{}\!\mathrm{d}\log\alpha_t}{\mathop{}\!\mathrm{d} t}
=
\frac{\mathop{}\!\mathrm{d}\log\alpha_\lambda}{\mathop{}\!\mathrm{d}\lambda}
\frac{\mathop{}\!\mathrm{d}\lambda_t}{\mathop{}\!\mathrm{d} t},
$$

得到对数信噪比时间下的 PF-ODE：

<span id="eq-diffusion-ode-lambda-eps"></span>

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d}\hat{{\mathbf{x}}}_\lambda}{\mathop{}\!\mathrm{d}\lambda}
=
\frac{\mathop{}\!\mathrm{d}\log\alpha_\lambda}{\mathop{}\!\mathrm{d}\lambda}
\hat{{\mathbf{x}}}_\lambda
-
\sigma_\lambda
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}
(\hat{{\mathbf{x}}}_\lambda,\lambda).
\end{aligned}
\tag{9.4.4}
$$

对[式 (9.4.4)](04-dpm-solver.md#eq-diffusion-ode-lambda-eps)应用相同的指数积分（常数变易）技术，得到以下精确解。

<span id="exp-wgt-exact-sol"></span>

**命题 9.4.1：指数加权精确解**

给定初值 ${\mathbf{x}}_s$，对应时刻 $s>0$，则精确经验 PF-ODE 解 $\widetilde{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)$ 在时刻 $t\in(0,s]$ 可重新表达为

<span id="eq-analytic-solution"></span>

$$
\widetilde{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
=
\frac{\alpha_t}{\alpha_s}{\mathbf{x}}_s
-
\alpha_t
\int_{\lambda_s}^{\lambda_t}
e^{-\lambda}
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda)
\mathop{}\!\mathrm{d}\lambda.
\tag{9.4.5}
$$

当端点满足 $\sigma_0=0$ 时，$t=0$ 的情况通过相应极限理解。

**命题证明。**

根据[式 (9.4.4)](04-dpm-solver.md#eq-diffusion-ode-lambda-eps)，线性项对应的积分因子满足

$$
\exp\left(
\int_{\lambda_s}^{\lambda_t}
\frac{\mathop{}\!\mathrm{d}\log\alpha_\lambda}{\mathop{}\!\mathrm{d}\lambda}
\mathop{}\!\mathrm{d}\lambda
\right)
=
\frac{\alpha_t}{\alpha_s}.
$$

因此，常数变易公式给出

$$
\begin{aligned}
\widetilde{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
&=
\frac{\alpha_t}{\alpha_s}{\mathbf{x}}_s
-
\int_{\lambda_s}^{\lambda_t}
\frac{\alpha_t}{\alpha_\lambda}
\sigma_\lambda
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda)
\mathop{}\!\mathrm{d}\lambda
\\
&=
\frac{\alpha_t}{\alpha_s}{\mathbf{x}}_s
-
\alpha_t
\int_{\lambda_s}^{\lambda_t}
e^{-\lambda}
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda)
\mathop{}\!\mathrm{d}\lambda,
\end{aligned}
$$

最后一个等号来自 $\sigma_\lambda/\alpha_\lambda=e^{-\lambda}$。

关键数值结果现在已经明确。在 $\lambda$ 时间下，精确解中唯一与模型有关的量出现在

$$
\int_{\lambda_s}^{\lambda_t}
e^{-\lambda}
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda)\mathop{}\!\mathrm{d}\lambda.
$$

当使用 $\lambda$ 中的 Taylor 多项式局部近似模型输出时，所得指数加权多项式项可以闭式积分，产生显式高阶更新系数。因此，与引言中的因子分解视角一样，已知调度依赖解析保留，近似则集中于神经网络预测。

**为什么重参数化时间？** 对于严格单调的 $\lambda(t)$，局部一阶变量替换给出

$$
|\Delta t|
\approx
\frac{|\Delta\lambda|}{|\lambda'(t)|}.
$$

因此，如果数值网格采用固定步长 $|\Delta\lambda|$，则诱导的 $t$ 步长在 $|\lambda'(t)|$ 较大处更小（即 $\lambda$ 随 $t$ 快速变化之处），在 $|\lambda'(t)|$ 较小处更大。

重参数化本身不改变 PF-ODE 的解路径，只改变同一路径的参数化方式：

$$
\frac{\mathop{}\!\mathrm{d} \hat{\mathbf{x}}_\lambda}{\mathop{}\!\mathrm{d} \lambda}
=
\frac{1}{\lambda'(t)}
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}_t}{\mathop{}\!\mathrm{d} t}.
$$

因此，在 $|\lambda'(t)|$ 较大处，轨迹相对于 $t$ 的同样变化，会在更大的 $\lambda$ 范围内表示。

更关键的是，$\lambda(t)$ 直接根据信号与噪声之间的相对平衡，度量扩散过程的进展。固定 $\Delta\lambda$ 对应于 $\alpha_t/\sigma_t$ 的固定乘性变化，等价于对数信噪比的固定加性变化；而相同的 $\Delta t$ 在不同时刻可能对应非常不同的扰动程度变化。因此，在 $\lambda$ 中均匀选择数值节点，是沿扩散轨迹重新分配分辨率的自然方式。原始时间变量 $t$ 中诱导的具体间距取决于所选噪声调度。两个简单例子可以说明这一影响：

-   **$(\alpha_t,\sigma_t)=(1-t,t)$。** 这对应流匹配常用的线性插值。此时

    $$
    \lambda(t)
    =
    \log\frac{1-t}{t},
    \qquad
    \lambda'(t)
    =
    -\frac{1}{t(1-t)},
    $$

    所以局部有

    $$
    |\Delta t|
    \approx
    |\Delta\lambda|\,t(1-t).
    $$

    因此，$\lambda$ 中的均匀网格，会在 $t$ 中的两个端点 $t\to0$ 和 $t\to1$ 附近诱导非常小的步长，在轨迹中部诱导更大的步长。

-   **$(\alpha_t,\sigma_t)=(1,t)$。** 当自变量 $t$ 直接取为噪声水平 $\sigma_t$ 时，这对应 EDM 参数化（[Karras et al., 2022](../../references.md#ref-karras2022elucidating)）。此时

    $$
    \lambda(t)
    =
    \log\frac{1}{t},
    \qquad
    \lambda'(t)
    =
    -\frac{1}{t},
    $$

    因此

    $$
    |\Delta t|
    \approx
    |\Delta\lambda|\,t.
    $$

    事实上，$\lambda$ 中的均匀间距意味着

    $$
    t=e^{-\lambda},
    $$

    所以对应的噪声水平按等比方式排列。因此，网格在低噪声水平（高信噪比）处的绝对间距更细，在高噪声水平（低信噪比）处更粗。

## 9.4.2 用 Taylor 展开估计积分 {#subsec-dpm-method-2}

DEIS 与 DPM-Solver 利用相同的因子分解原理，但获取高阶信息的方式不同。DEIS 复用先前已完成步骤的预测，将神经网络预测外推到下一区间。DPM-Solver 则在对数信噪比坐标 $\lambda$ 中工作，使用神经网络预测的局部 Taylor 展开，所需高阶信息来自当前步内的额外计算。下面推导这一单步构造。

根据[式 (9.4.5)](04-dpm-solver.md#eq-analytic-solution)，以当前数值状态 $\tilde{\mathbf{x}}_s$ 初始化的精确经验 PF-ODE 流为

<span id="eqn-analytic-solution-each-step"></span>

$$
\widetilde{\boldsymbol{\Psi}}_{s\to t}(\tilde{\mathbf{x}}_s)
=
\frac{\alpha_t}{\alpha_s}\tilde{\mathbf{x}}_s
-
\alpha_t
\int_{\lambda_s}^{\lambda_t}
e^{-\lambda}
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda)
\mathop{}\!\mathrm{d}\lambda,
\tag{9.4.6}
$$

其中 $\hat{\mathbf{x}}_{\lambda_s}=\tilde{\mathbf{x}}_s$，且 $\hat{\mathbf{x}}_\lambda$ 表示当前步上的相应精确经验 PF-ODE 轨迹。因此，在解析处理已知调度相关因子后，剩余数值问题是在加权积分内近似

$$
\lambda
\longmapsto
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda)
$$

对于 $n\geq1$，将这一神经网络预测在 $\lambda_s$ 附近展开：

$$
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda)
=
\sum_{k=0}^{n-1}
\frac{(\lambda-\lambda_s)^k}{k!}
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}^{(k)}
(\hat{\mathbf{x}}_{\lambda_s},\lambda_s)
+
\mathcal O\bigl((\lambda-\lambda_s)^n\bigr),
$$

其中

$$
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}^{(k)}
(\hat{\mathbf{x}}_\lambda,\lambda)
:=
\frac{\mathop{}\!\mathrm{d}^k}{\mathop{}\!\mathrm{d}\lambda^k}
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda)
$$

表示*沿经验 PF-ODE 轨迹的* $k$ 阶*全导数*。特别地，

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d}\lambda}
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda)
=
\frac{\partial}{\partial\lambda}
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda)
+
J_{{\mathbf{x}}}
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda)
\frac{\mathop{}\!\mathrm{d}\hat{\mathbf{x}}_\lambda}{\mathop{}\!\mathrm{d}\lambda},
$$

因此，该导数既包含对 $\lambda$ 的显式依赖，也包含经演化状态产生的依赖。

将 Taylor 展开代入[式 (9.4.6)](04-dpm-solver.md#eqn-analytic-solution-each-step)，得到

<span id="eq-k-th-expansion"></span>

$$
\widetilde{\boldsymbol{\Psi}}_{s\to t}(\tilde{\mathbf{x}}_s)
=
\frac{\alpha_t}{\alpha_s}\tilde{\mathbf{x}}_s
-
\alpha_t
\sum_{k=0}^{n-1}
{\color{cyan}
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}^{(k)}
(\hat{\mathbf{x}}_{\lambda_s},\lambda_s)}
\,{\color{orange}C_k}
+
\mathcal O(h^{n+1}),
\tag{9.4.7}
$$

其中

$$
h:=\lambda_t-\lambda_s,
\qquad
{\color{orange}C_k}
:=
\int_{\lambda_s}^{\lambda_t}
e^{-\lambda}
\frac{(\lambda-\lambda_s)^k}{k!}
\mathop{}\!\mathrm{d}\lambda.
$$

系数 ${\color{orange}C_k}$ 只依赖对数信噪比区间，可以解析计算。

如果能够获得所需轨迹导数，去掉 $\mathcal O(h^{n+1})$ 余项就得到相应形式的 $n$ 阶更新。DPM-Solver 通过额外模型计算近似这些导数，避免显式对神经网络求导，下一小节将说明。

**示例：DPM-Solver-1**

当 $n=1$ 时，精确经验流满足

$$
\widetilde{\boldsymbol{\Psi}}_{s\to t}(\tilde{\mathbf{x}}_s)
=
\frac{\alpha_t}{\alpha_s}\tilde{\mathbf{x}}_s
-
\sigma_t
\left(e^h-1\right)
\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s)
+
\mathcal O(h^2).
$$

去掉 $\mathcal O(h^2)$ 余项，定义 DPM-Solver-1：

<span id="eq-dpm-1"></span>

$$
\begin{aligned}
\tilde{\mathbf{x}}_t^{\mathrm{DPM\text{-}Solver\text{-}1}}
:=
\frac{\alpha_t}{\alpha_s}\tilde{\mathbf{x}}_s
-
\sigma_t
\left(e^h-1\right)
\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s).
\end{aligned}
\tag{9.4.8}
$$

这一有限步更新恰好是确定性 DDIM 更新，见[第 9.4.4 节](04-dpm-solver.md#subsec-ddim-dpm-solver-1)。

DPM-Solver-$n$ 在 $n \geq 2$ 时需要计算 $k^{\text{th}}$ 导数 ${\color{cyan}\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}^{(k)}(\hat{\mathbf{x}}_\lambda, \lambda)}$，其中 $k \leq n-1$。但实践中直接计算高阶导数开销很大。[Lu et al. (2022)](../../references.md#ref-lu2022dpm) 也为这些导数提出了高效近似方法，下一小节将详细介绍。

## 9.4.3 DPM-Solver-$n$ 的实现 {#subsec-dpm-n-method}

**DPM-Solver-$n$，其中 $n \geq 2$。** 实践中，实现高阶 DPM-Solver-$n$ 需要：

-   预先计算系数 ${\color{orange}C_k}$；

-   近似 $k^{\text{th}}$ 导数 ${\color{cyan}\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}^{(k)}(\hat{\mathbf{x}}_\lambda,\lambda)}$，其中 $k \leq n-1$，以绕过精确高阶导数的高昂计算开销；这一挑战在 ODE 文献中已得到充分研究（[Hochbruck and Ostermann, 2005](../../references.md#ref-hochbruck2005explicit); [Luan, 2021](../../references.md#ref-luan2021efficient)）。一种常见策略是有限差分近似。

下面详细说明这两点。

**预先计算 ${\color{orange}C_k}$。** 令 $s$ 和 $t$ 分别表示起始与终止时刻，并定义 $h:=\lambda_t-\lambda_s$。精确经验 PF-ODE 流的 Taylor 展开具有如下形式

<span id="eq-analytic-expansion"></span>

$$
\widetilde{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
=
\frac{\alpha_t}{\alpha_s}{\mathbf{x}}_s
-
\sigma_t
\sum_{k=0}^{n-1}
h^{k+1}
\varphi_{k+1}(h)
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}^{(k)}
(\hat{\mathbf{x}}_{\lambda_s},\lambda_s)
+
\mathcal O(h^{n+1}),
\tag{9.4.9}
$$

其中，对于 $m\geq1$，

$$
\varphi_m(h)
:=
\frac{
e^h-\sum_{j=0}^{m-1}\frac{h^j}{j!}
}{
h^m
},
\qquad
\varphi_m(0):=\frac{1}{m!}.
$$

前三种情况为

$$
\begin{aligned}
\varphi_1(h)
&=
\frac{e^h-1}{h},
&
\varphi_2(h)
&=
\frac{e^h-h-1}{h^2},
&
\varphi_3(h)
&=
\frac{e^h-\frac{h^2}{2}-h-1}{h^3}.
\end{aligned}
$$

**示例：使用精确导数的三阶展开**

当 $n=3$ 时，[式 (9.4.9)](04-dpm-solver.md#eq-analytic-expansion)变为

<span id="eq-example-dpm-solver-3-exact"></span>

$$
\begin{aligned}
\widetilde{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
=
\frac{\alpha_t}{\alpha_s}{\mathbf{x}}_s
&-
\sigma_t
\left(e^h-1\right)
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_{\lambda_s},\lambda_s)
\\
&-
\sigma_t
\left(e^h-h-1\right)
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}^{(1)}
(\hat{\mathbf{x}}_{\lambda_s},\lambda_s)
\\
&-
\sigma_t
\left(
e^h-\frac{h^2}{2}-h-1
\right)
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}^{(2)}
(\hat{\mathbf{x}}_{\lambda_s},\lambda_s)
+
\mathcal O(h^4).
\end{aligned}
\tag{9.4.10}
$$

**近似 ${\color{cyan}\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}^{(k)}(\hat{\mathbf{x}}_\lambda,\lambda)}$，其中 $k \leq n-1$。** 对于 $n \geq 2$，按照单步 ODE 求解器的标准方法（[Atkinson et al., 2009](../../references.md#ref-atkinson2009numerical)），[Lu et al. (2022)](../../references.md#ref-lu2022dpm) 在当前步内增加模型计算，以近似高阶导数。下面以 $n=2$ 为例说明。

令 $\gamma\in(0,1]$ 指定对数信噪比区间 $[\lambda_s,\lambda_t]$ 中的一个阶段点。给定估计 $\tilde{\mathbf{x}}_s$，对应位置为 $s$，定义

$$
s^{\mathrm{mid}}
=
t_\lambda\left(\lambda_s+\gamma h\right),
\qquad
h:=\lambda_t-\lambda_s.
$$

当 $\gamma\in(0,1)$ 时，这是内部点；$\gamma=1$ 则对应端点 $t$。先使用一次 DPM-Solver-1 更新预测该阶段的状态：

$$
{\mathbf{x}}^{\mathrm{mid}}
=
\frac{\alpha_{s^{\mathrm{mid}}}}{\alpha_s}\tilde{\mathbf{x}}_s
-
\sigma_{s^{\mathrm{mid}}}
\left(e^{\gamma h}-1\right)
\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s).
$$

随后，起点与阶段点处的模型预测之差，提供沿轨迹一阶全导数的有限差分近似：

$$
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}^{(1)}
(\hat{\mathbf{x}}_{\lambda_s},\lambda_s)
\approx
\frac{
\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}
({\mathbf{x}}^{\mathrm{mid}},s^{\mathrm{mid}})
-
\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}
(\tilde{\mathbf{x}}_s,s)
}{
\gamma h
}.
$$

将此近似代入二阶展开，得到

<span id="eq-dpm-solver-2-app"></span>

$$
\begin{aligned}
\widetilde{\boldsymbol{\Psi}}_{s\to t}(\tilde{\mathbf{x}}_s)
=
\frac{\alpha_t}{\alpha_s}\tilde{\mathbf{x}}_s
&-
\sigma_t
\left(e^h-1\right)
\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s)
\\
&-
\frac{\sigma_t}{\gamma h}
\left(e^h-h-1\right)
\left[
\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}
({\mathbf{x}}^{\mathrm{mid}},s^{\mathrm{mid}})
-
\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}
(\tilde{\mathbf{x}}_s,s)
\right]
+
\mathcal O(h^3).
\end{aligned}
\tag{9.4.11}
$$

因此，去掉 $\mathcal O(h^3)$ 余项，就得到二阶单步更新。

对于中点选择 $\gamma=\tfrac12$，DPM-Solver-2 可写为[算法 5](04-dpm-solver.md#alg-dpm-solver-2)中更简单的两阶段形式。这一更新与[式 (9.4.11)](04-dpm-solver.md#eq-dpm-solver-2-app)只相差 $\mathcal O(h^3)$，因此具有相同的二阶精度。

<section class="book-algorithm" id="alg-dpm-solver-2" aria-labelledby="alg-dpm-solver-2-title" markdown="1">

<div class="book-algorithm__title" id="alg-dpm-solver-2-title" markdown="span">
算法 5：DPM-Solver-2（$\gamma=\tfrac12$）。
</div>

<div class="book-algorithm__meta" markdown="1">

**输入：** 初始值 ${\mathbf{x}}_T$、时间步 $\{t_i\}_{i=0}^M$、模型 ${\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}$

</div>

<div class="book-algorithm__scroll" tabindex="0" role="region" aria-label="算法 5 的步骤" markdown="1">
<div class="book-algorithm__lines" markdown="1">

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">1</div>
<div class="book-algorithm__statement" markdown="1">

$\tilde{\mathbf{x}}_{t_0}\gets{\mathbf{x}}_T$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">2</div>
<div class="book-algorithm__statement" markdown="1">

**for** $i\gets 1$ to $M$ **do**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">3</div>
<div class="book-algorithm__statement" markdown="1">

$h_i \gets \lambda_{t_i}-\lambda_{t_{i-1}}$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">4</div>
<div class="book-algorithm__statement" markdown="1">

$s_i^{\mathrm{mid}} \gets t_\lambda\left( \tfrac{\lambda_{t_{i-1}}+\lambda_{t_i}}{2} \right)$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">5</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{x}}_i^{\mathrm{mid}} \gets \tfrac{\alpha_{s_i^{\mathrm{mid}}}}{\alpha_{t_{i-1}}} \tilde{\mathbf{x}}_{t_{i-1}} - \sigma_{s_i^{\mathrm{mid}}} \left(e^{h_i/2}-1\right) {\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times} (\tilde{\mathbf{x}}_{t_{i-1}},t_{i-1})$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">6</div>
<div class="book-algorithm__statement" markdown="1">

$\tilde{\mathbf{x}}_{t_i} \gets \tfrac{\alpha_{t_i}}{\alpha_{t_{i-1}}} \tilde{\mathbf{x}}_{t_{i-1}} - \sigma_{t_i} \left(e^{h_i}-1\right) {\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times} ({\mathbf{x}}_i^{\mathrm{mid}},s_i^{\mathrm{mid}})$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">7</div>
<div class="book-algorithm__statement" markdown="1">

**end for**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">8</div>
<div class="book-algorithm__statement" markdown="1">

**return** $\tilde{\mathbf{x}}_{t_M}$

</div>
</div>

</div>
</div>

</section>

**说明。**

上述有限差分近似神经网络预测沿经验 PF-ODE 轨迹的 $\lambda$ *全导数*。在标准光滑性假设下，

$$
\frac{
\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}
({\mathbf{x}}^{\mathrm{mid}},s^{\mathrm{mid}})
-
\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}
(\tilde{\mathbf{x}}_s,s)
}{
\gamma h
}
=
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}^{(1)}
(\hat{\mathbf{x}}_{\lambda_s},\lambda_s)
+
\mathcal O(h).
$$

这里，中间状态本身由一阶 DPM-Solver-1 步得到，其阶段误差为 $\mathcal O(h^2)$；在模型光滑时，这不改变差商的 $\mathcal O(h)$ 精度。

在[式 (9.4.11)](04-dpm-solver.md#eq-dpm-solver-2-app)中，这一导数近似乘以

$$
e^h-h-1=\mathcal O(h^2),
$$

因此诱导的误差为 $\mathcal O(h^3)$。所以，对任何固定 $\gamma\in(0,1]$，所得方法都具有二阶精度。

这一单步构造使用两次模型计算：一次在 $(\tilde{\mathbf{x}}_s,s)$，一次在预测阶段 $({\mathbf{x}}^{\mathrm{mid}},s^{\mathrm{mid}})$。$\gamma$ 的选择不改变形式上的二阶精度，但会改变阶段位置及误差常数。特别地，$\gamma=\tfrac12$ 将第二次计算放在对数信噪比区间的中点，得到常用的 DPM-Solver-2 中点版本。

**实现细节：采样时间步的选择。** 由于 DPM-Solver 在对数信噪比时间 $\lambda$ 中构造局部高阶近似，一个自然选择是在 $\lambda$ 中均匀放置数值节点。对于有限终止时间 $t_M>0$，得到

$$
\lambda_{t_i}
=
\lambda_{t_0}
+
\frac{i}{M}
\left(
\lambda_{t_M}-\lambda_{t_0}
\right),
\qquad
i=0,\ldots,M.
$$

如果 $\sigma_0=0$，则 $\lambda_0=+\infty$，所以实践中使用对应于小正时间的有限终端对数信噪比，端点则通过相应极限或最终去噪步骤处理。

虽然推导在 $\lambda$ 空间进行，预训练模型与调度 $(\alpha_t,\sigma_t)$ 通常仍以原始扩散时间 $t$ 参数化。因此，采样时，每当需要计算模型或调度系数，就通过下式将所选对数信噪比节点映射回去

$$
t_i=t_\lambda(\lambda_{t_i})
$$

例如见[算法 5](04-dpm-solver.md#alg-dpm-solver-2)。因此，$\lambda$ 作为数值积分坐标，而采样器仍在对应时间标签处计算原始预训练扩散模型。

## 9.4.4 DDIM $=$ DPM-Solver-1 {#subsec-ddim-dpm-solver-1}

一阶 DPM-Solver 精确恢复 DDIM。两种方法都从同一噪声预测 PF-ODE 的精确积分表示出发，并在一步内作出相同近似：

$$
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)
\approx
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s).
$$

随后解析积分剩余的调度相关项。DDIM 直接在前文推导的调度适配形式中应用这一构造，DPM-Solver 则先使用对数信噪比坐标改写积分

$$
\lambda_t
=
\log\frac{\alpha_t}{\sigma_t}.
$$

在一阶情况下，改变数值时钟并不改变模型预测的常数近似。因此，两种推导产生完全相同的有限步更新。

<span id="ddim-dpm"></span>

**命题 9.4.2：DDIM 就是 DPM-Solver-1**

对于固定扩散调度 $(\alpha_t,\sigma_t)$，[式 (9.2.1)](02-ddim.md#eq-ddim-euler)中的确定性 DDIM 更新，与[式 (9.4.8)](04-dpm-solver.md#eq-dpm-1)中的 DPM-Solver-1 更新完全相同。

**命题证明。**

根据 $\lambda$ 的定义，

$$
\frac{\sigma_s}{\alpha_s}
=
e^{-\lambda_s},
\qquad
\frac{\sigma_t}{\alpha_t}
=
e^{-\lambda_t}.
$$

令

$$
h
=
\lambda_t-\lambda_s.
$$

则

$$
e^h
=
\frac{\alpha_t/\sigma_t}{\alpha_s/\sigma_s}
=
\frac{\alpha_t\sigma_s}{\sigma_t\alpha_s},
$$

因此

$$
-\sigma_t(e^h-1)
=
\sigma_t
-
\frac{\alpha_t}{\alpha_s}\sigma_s
=
\alpha_t
\left(
\frac{\sigma_t}{\alpha_t}
-
\frac{\sigma_s}{\alpha_s}
\right).
$$

将此恒等式代入 DPM-Solver-1 更新，得到

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

这恰好是[式 (9.2.1)](02-ddim.md#eq-ddim-euler)中的确定性 DDIM 更新。

因此，DPM-Solver-1 与 DDIM 是同一一阶有限步更新的两种推导。对数信噪比坐标在超过一阶后才产生实质影响：高阶 DPM-Solver 用它来近似噪声预测沿轨迹的变化，不再将预测在整个步骤中视为常数。

## 9.4.5 DPM-Solver 与 Runge–Kutta 型求解器的关系 {#subsec-dpm-rk}

DPM-Solver-1 恰好就是 DDIM 更新（[第 9.4.4 节](04-dpm-solver.md#subsec-ddim-dpm-solver-1)），因此其一阶行为已经明确。高阶问题在于：DPM-Solver 如何改进这一只计算一次模型的更新？

这里的联系是 Runge–Kutta（RK）方法。Euler 只使用该步起点的向量场，而 RK 方法通过*阶段*获取额外信息，即在当前步内生成的状态与时刻上进行计算；见[第 A.1.5 节](../../appendices/a-differential-equations/01-ordinary-differential-equations.md#app-de-solver-examples)。高阶 DPM-Solver 遵循同一原理：引入额外步内模型计算来捕捉神经网络预测的变化，而非在整个步骤中固定一次预测。扩散特有的区别在于，已知调度相关因子解析保留，分阶段近似只作用于学习到的预测。

**DPM-Solver-2 与普通中点 RK2。** 首先，将 DPM-Solver-2 与直接应用于[式 (9.4.4)](04-dpm-solver.md#eq-diffusion-ode-lambda-eps)中完整对数信噪比时间噪声预测 PF-ODE 的经典中点 RK2 比较。为简洁，将完整右端项记为 ${\mathbf{F}}(\hat{\mathbf{x}}_\lambda,\lambda)$。

对于中点选择 $\gamma=\tfrac12$，令

$$
\lambda_{\mathrm{mid}}
:=
\lambda_{s^{\mathrm{mid}}}
=
\lambda_s+\frac{h}{2},
\qquad
\alpha_{\mathrm{mid}}
:=
\alpha_{s^{\mathrm{mid}}},
\qquad
\sigma_{\mathrm{mid}}
:=
\sigma_{s^{\mathrm{mid}}},
$$

并简记

$$
\hat{\boldsymbol{\epsilon}}_s
:=
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,\lambda_s)
=
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s).
$$

经典中点 RK2 给出

$$
{\mathbf{x}}_{\mathrm{mid}}^{\mathrm{RK}}
=
\tilde{\mathbf{x}}_s
+
\frac{h}{2}
{\mathbf{F}}(\tilde{\mathbf{x}}_s,\lambda_s),
$$

随后

$$
\tilde{\mathbf{x}}_t^{\mathrm{RK2}}
=
\tilde{\mathbf{x}}_s
+
h\,
{\mathbf{F}}
\left(
{\mathbf{x}}_{\mathrm{mid}}^{\mathrm{RK}},
\lambda_{\mathrm{mid}}
\right).
$$

因此，普通 RK2 将两阶段构造应用于完整 PF-ODE 速度，其中同时包含已知调度相关项与神经网络预测。

[算法 5](04-dpm-solver.md#alg-dpm-solver-2)中的中点 DPM-Solver-2 则先构造阶段

$$
{\mathbf{x}}_{\mathrm{mid}}^{\mathrm{DPM2}}
=
\frac{\alpha_{\mathrm{mid}}}{\alpha_s}\tilde{\mathbf{x}}_s
-
\sigma_{\mathrm{mid}}
\left(
e^{h/2}-1
\right)
\hat{\boldsymbol{\epsilon}}_s.
$$

将相应中点预测记为

$$
\hat{\boldsymbol{\epsilon}}_{\mathrm{mid}}^{\mathrm{DPM2}}
:=
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}
\left(
{\mathbf{x}}_{\mathrm{mid}}^{\mathrm{DPM2}},
\lambda_{\mathrm{mid}}
\right),
$$

完整更新为

$$
\tilde{\mathbf{x}}_t^{\mathrm{DPM2}}
=
\frac{\alpha_t}{\alpha_s}\tilde{\mathbf{x}}_s
-
\sigma_t
\left(
e^h-1
\right)
\hat{\boldsymbol{\epsilon}}_{\mathrm{mid}}^{\mathrm{DPM2}}.
$$

结构差异十分清楚。普通中点 RK2 离散化完整 PF-ODE 速度，DPM-Solver-2 则解析保留已知扩散相关演化，只用两次模型计算近似噪声预测的变化。

在标准光滑性假设下，两者都是针对同一 $\lambda$ 时间经验 PF-ODE 的二阶方法。中间状态一般相差 $\mathcal O(h^2)$，因此完整单步更新满足

$$
\tilde{\mathbf{x}}_t^{\mathrm{DPM2}}
-
\tilde{\mathbf{x}}_t^{\mathrm{RK2}}
=
\mathcal O(h^3).
$$

所以，DPM-Solver-2 遵循与经典 RK2 相同的两阶段中点原理，同时在应用数值近似前利用已知扩散结构。

**DPM-Solver-2 与指数中点 RK2。** 更接近的比较是显式保留[式 (9.4.4)](04-dpm-solver.md#eq-diffusion-ode-lambda-eps)中的半线性分解，并解析处理已知线性部分。相应的*指数中点 RK2* 构造，先用指数 Euler 阶段到达中点：

$$
{\mathbf{x}}_{\mathrm{mid}}^{\mathrm{ExpRK}}
=
\frac{\alpha_{\mathrm{mid}}}{\alpha_s}\tilde{\mathbf{x}}_s
-
\alpha_{\mathrm{mid}}\sigma_s
\left[
\int_{\lambda_s}^{\lambda_{\mathrm{mid}}}
\frac{1}{\alpha_\lambda}\mathop{}\!\mathrm{d}\lambda
\right]
\hat{\boldsymbol{\epsilon}}_s.
$$

计算

$$
\hat{\boldsymbol{\epsilon}}_{\mathrm{mid}}^{\mathrm{ExpRK}}
:=
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}
\left(
{\mathbf{x}}_{\mathrm{mid}}^{\mathrm{ExpRK}},
\lambda_{\mathrm{mid}}
\right),
$$

后，完整指数中点更新为

$$
\tilde{\mathbf{x}}_t^{\mathrm{ExpRK2}}
=
\frac{\alpha_t}{\alpha_s}\tilde{\mathbf{x}}_s
-
\alpha_t\sigma_{\mathrm{mid}}
\left[
\int_{\lambda_s}^{\lambda_t}
\frac{1}{\alpha_\lambda}\mathop{}\!\mathrm{d}\lambda
\right]
\hat{\boldsymbol{\epsilon}}_{\mathrm{mid}}^{\mathrm{ExpRK}}.
$$

当线性系数为常数时，这些积分系数化为显式指数中点 RK2 通常的 $\varphi_1$ 系数。

DPM-Solver-2 使用相同的中点阶段原理，但利用更细的噪声预测分解

$$
-\sigma_\lambda
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\hat{\mathbf{x}}_\lambda,\lambda).
$$

由于 $\sigma_\lambda$ 同样由扩散调度确定，DPM-Solver-2 将其保留在积分内，只对神经网络预测进行分阶段近似。因此，其中点与完整步骤系数分别为

$$
\alpha_{\mathrm{mid}}
\int_{\lambda_s}^{\lambda_{\mathrm{mid}}}
\frac{\sigma_\lambda}{\alpha_\lambda}\mathop{}\!\mathrm{d}\lambda
=
\sigma_{\mathrm{mid}}
\left(e^{h/2}-1\right),
\quad \text{and}\quad
\alpha_t
\int_{\lambda_s}^{\lambda_t}
\frac{\sigma_\lambda}{\alpha_\lambda}\mathop{}\!\mathrm{d}\lambda
=
\sigma_t
\left(e^h-1\right),
$$

从而恢复[算法 5](04-dpm-solver.md#alg-dpm-solver-2)中的更新。

因此，三种构造的区别在于 RK 阶段近似什么：

$$
\begin{array}{rl}
\text{\textbf{Plain RK2:} }
&
\text{stage the complete PF-ODE velocity},
\\[4pt]
\textbf{\textbf{ExpRK2:} }
&
\begin{aligned}[t]
&\text{propagate the linear part exactly;}\\
&\text{stage the coefficient-weighted residual }
\sigma_\lambda\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times},
\end{aligned}
\\[6pt]
\text{\textbf{DPM-Solver-2:} }
&
\text{also retain }\sigma_\lambda\text{ analytically and stage only }
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}.
\end{array}
$$

指数 RK 与 DPM 中点阶段的差异为

$$
{\mathbf{x}}_{\mathrm{mid}}^{\mathrm{DPM2}}
-
{\mathbf{x}}_{\mathrm{mid}}^{\mathrm{ExpRK}}
=
-
\alpha_{\mathrm{mid}}
\left[
\int_{\lambda_s}^{\lambda_{\mathrm{mid}}}
\frac{\sigma_\lambda-\sigma_s}{\alpha_\lambda}
\mathop{}\!\mathrm{d}\lambda
\right]
\hat{\boldsymbol{\epsilon}}_s
=
\mathcal O(h^2).
$$

对于完整步骤，在 $\lambda_{\mathrm{mid}}$ 附近进行 Taylor 展开，得到

$$
\int_{\lambda_s}^{\lambda_t}
\frac{\sigma_\lambda-\sigma_{\mathrm{mid}}}{\alpha_\lambda}
\mathop{}\!\mathrm{d}\lambda
=
\mathcal O(h^3),
$$

而神经网络预测的光滑性给出

$$
\hat{\boldsymbol{\epsilon}}_{\mathrm{mid}}^{\mathrm{DPM2}}
-
\hat{\boldsymbol{\epsilon}}_{\mathrm{mid}}^{\mathrm{ExpRK}}
=
\mathcal O(h^2).
$$

因此，

$$
\tilde{\mathbf{x}}_t^{\mathrm{DPM2}}
-
\tilde{\mathbf{x}}_t^{\mathrm{ExpRK2}}
=
\mathcal O(h^3).
$$

所以 DPM-Solver-2 与指数中点 RK2 一致到二阶，但 DPM-Solver-2 在加权积分内多保留了一个解析已知量 $\sigma_\lambda$。

**说明。**

上述讨论针对 DPM-Solver 推导所用的噪声预测表述。如果预训练模型采用其他预测参数化，可以在每个阶段通过[式 (6.3.1)](../06-unified-perspective/03-equivalent-parameterizations.md#eq-predictions-equivalence)将输出逐点转换为噪声预测。与[第 9.2.2 节](02-ddim.md#subsec-ddim-classical-euler)中的 DDIM 情况一样，不应将其与直接对原始预测场应用经典 RK 方法混淆：逐点转换与有限步离散化是不同操作。

**DPM-Solver 与 Runge–Kutta 方法**

DPM-Solver 将 Runge–Kutta 方法的步内阶段原理，与已知扩散动力学的解析处理结合起来。DPM-Solver-1 恰好就是 DDIM，高阶 DPM-Solver 则通过额外步内计算捕捉神经网络预测的变化。与经典 RK 方法的联系体现在结构上，并不意味着有限步公式相等：DPM-Solver 在近似学习预测之前，先解析保留已知扩散相关权重。
