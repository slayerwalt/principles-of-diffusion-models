# 8.4 免训练引导 {#sec-training-free-guidance}

免训练引导使用外部条件、损失或测量目标，修改固定预训练扩散模型的采样动力学，无需重新训练扩散模型本身。[式 (8.1.1)](01-prologue.md#eq-guidance-bayes)中的 Bayes 分解在 score 空间提供了有用的动机：引导在无条件 score 上加入依赖条件的方向。实践中，可以利用任务特定损失，对预测的干净样本或采样使用的 score 构造修正。

我们首先在[第 8.4.1 节](04-training-free-guidance.md#subsec-tfg)介绍这两种一般形式，再在[第 8.4.2 节](04-training-free-guidance.md#subsec-guided-marginal-path)讨论这类代理引导动力学实际生成什么分布，最后在[第 8.4.3 节](04-training-free-guidance.md#subsec-inverse-problem)介绍用于逆问题的代表性免训练方法。

**设定与记号。** 令 ${\mathbf{c}}$ 表示条件变量。假设可以使用预训练扩散模型 ${\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t,t)$，其表达形式为 score 预测[^1]。另外，假设给定任务损失

$$
\ell(\cdot,{\mathbf{c}})\colon\mathbb{R}^D\to\mathbb{R}
$$

用于度量与所需条件的不一致程度，因此 $\ell({\mathbf{x}},{\mathbf{c}})$ 越小表示越一致。例如，与参考目标的感知距离，或由 CLIP 等预训练表示（[Radford et al., 2021](../../references.md#ref-radford2021learning)）导出的特征空间距离或负相似度。

考虑标准的线性 Gaussian 前向加噪核

$$
p_t(\cdot|{\mathbf{x}}_0)
:=
\mathcal N\bigl(\cdot;\alpha_t{\mathbf{x}}_0,\sigma_t^2{\mathbf{I}}\bigr).
$$

我们借用后文[式 (9.2.2)](../09-numerical-solvers/02-ddim.md#eq-ddim-update-all)中介绍的 DDIM 更新作为具体例子：

<span id="eq-ddim-update-clean-noise-score"></span>

$$
\begin{aligned}
{\mathbf{x}}_{t\rightarrow t-1}
=
\alpha_{t-1}
\underbrace{\hat{{\mathbf{x}}}_0({\mathbf{x}}_t)}_{\text{in data space}}
-
\sigma_{t-1}\sigma_t
\underbrace{\hat{{\mathbf{s}}}({\mathbf{x}}_t)}_{\text{in noise space}},
\end{aligned}
\tag{8.4.1}
$$

其中，$\hat{{\mathbf{x}}}_0({\mathbf{x}}_t):={\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t,t)$ 是干净的 ${\mathbf{x}}$ 预测，$\hat{{\mathbf{s}}}({\mathbf{x}}_t):={\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t,t)$ 是噪声水平 $t$ 下的 score 预测。

## 8.4.1 免训练引导的概念框架 {#subsec-tfg}

一种常见策略是使用任务损失的梯度，修改采样更新中的干净样本预测或 score 项。为区分这两种操作，定义数据空间方向

$$
\mathcal G_t^{\mathrm{data}}({\mathbf{x}}_t,{\mathbf{c}})
:=
-
\left.
\nabla_{{\mathbf{x}}_0}\ell({\mathbf{x}}_0,{\mathbf{c}})
\right|_{{\mathbf{x}}_0=\hat{{\mathbf{x}}}_0({\mathbf{x}}_t)}.
$$

对于噪声空间修正，沿用引言中介绍的引导方向，

$$
\mathcal G_t({\mathbf{x}}_t,{\mathbf{c}})
:=
\nabla_{{\mathbf{x}}_t}\log{\mathbf{h}}_t({\mathbf{x}}_t;{\mathbf{c}}),
$$

其中正值引导势函数 ${\mathbf{h}}_t$ 将在下文指定。

利用[式 (8.4.1)](04-training-free-guidance.md#eq-ddim-update-clean-noise-score)中的 DDIM 更新，两种修正可紧凑地写为

<span id="eq-ddim-update-clean-noise-guide"></span>

$$
\begin{aligned}
{\mathbf{x}}_{t\rightarrow t-1}
=
\alpha_{t-1}
\underbrace{\left(
\hat{{\mathbf{x}}}_0({\mathbf{x}}_t)
+
{\color{orange}
\eta_t^{\text{data}}
\mathcal G_t^{\text{data}}({\mathbf{x}}_t,{\mathbf{c}})}
\right)}_{\text{A. data space}}
-
\sigma_{t-1}\sigma_t
\underbrace{\left(
\hat{{\mathbf{s}}}({\mathbf{x}}_t)
+
{\color{orange}
\eta_t^{\text{latent}}
\mathcal G_t({\mathbf{x}}_t,{\mathbf{c}})}
\right)}_{\text{B. noise space}},
\end{aligned}
\tag{8.4.2}
$$

其中，$\eta_t^{\text{data}},\eta_t^{\text{latent}}\geq0$ 分别控制数据空间与噪声空间修正的强度。

**A. 数据空间中的引导。** 数据空间修正通过执行一个降低任务损失的局部步骤，直接改进当前干净样本估计：

$$
\hat{{\mathbf{x}}}_0({\mathbf{x}}_t)
\;\longmapsto\;
\hat{{\mathbf{x}}}_0({\mathbf{x}}_t)
+
\eta_t^{\text{data}}
\mathcal G_t^{\text{data}}({\mathbf{x}}_t,{\mathbf{c}}).
$$

因此，先将干净预测移向更符合条件的位置，再用于后续采样更新。需要多个优化步骤时，也可以反复应用这一修正。

代表性例子包括 MGPD（[He et al., 2023](../../references.md#ref-he2023manifold)）和 UGD（[Bansal et al., 2023](../../references.md#ref-bansal2023universal)）。

**B. 噪声空间中的引导。** 一种密切相关的策略是直接对采样中使用的 score 施加引导。沿用[第 8.1 节](01-prologue.md#sec-prologue-guidance)介绍的势函数视角，一种常见构造是在当前去噪估计上计算任务损失，并选择

$$
{\mathbf{h}}_t({\mathbf{x}}_t;{\mathbf{c}})
:=
\exp\!\left(
-\tau\,\ell\bigl(\hat{{\mathbf{x}}}_0({\mathbf{x}}_t),{\mathbf{c}}\bigr)
\right),
\qquad \tau>0.
$$

根据上面对 $\mathcal G_t$ 的定义，得到

$$
\begin{aligned}
\mathcal G_t({\mathbf{x}}_t,{\mathbf{c}})
=
\nabla_{{\mathbf{x}}_t}\log{\mathbf{h}}_t({\mathbf{x}}_t;{\mathbf{c}})
=
-\tau
\nabla_{{\mathbf{x}}_t}
\ell\bigl(\hat{{\mathbf{x}}}_0({\mathbf{x}}_t),{\mathbf{c}}\bigr).
\end{aligned}
$$

因此，[式 (8.4.2)](04-training-free-guidance.md#eq-ddim-update-clean-noise-guide)中的 score 项变为

$$
\hat{{\mathbf{s}}}({\mathbf{x}}_t)
+
\eta_t^{\text{latent}}
\mathcal G_t({\mathbf{x}}_t,{\mathbf{c}}).
$$

将这一实际更新与[式 (8.1.3)](01-prologue.md#eq-generic-guided-score)中的总体构造比较可见，$\eta_t^{\text{latent}}$ 起到一般引导强度 $w_t$ 的作用，而 $\hat{{\mathbf{s}}}({\mathbf{x}}_t)$ 用学习到的估计替代精确无条件 score。

当 ${\mathbf{h}}_t({\mathbf{x}}_t;{\mathbf{c}})$ 与真实条件似然 $p_t({\mathbf{c}}|{\mathbf{x}}_t)$ 成正比，且比例因子不依赖 ${\mathbf{x}}_t$ 时，$\mathcal G_t({\mathbf{x}}_t,{\mathbf{c}})$ 恢复 Bayes 引导方向 $\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{c}}|{\mathbf{x}}_t)$。对于一般任务损失，$\mathcal G_t$ 则定义了代理引导方向。

计算 $\mathcal G_t$ 需要通过 ${\mathbf{x}}$ 预测进行求导：

$$
\mathcal G_t({\mathbf{x}}_t,{\mathbf{c}})
=
-\tau
\left[
\nabla_{{\mathbf{x}}_t}\hat{{\mathbf{x}}}_0({\mathbf{x}}_t)
\right]^\top
\left.
\nabla_{{\mathbf{x}}_0}\ell({\mathbf{x}}_0,{\mathbf{c}})
\right|_{{\mathbf{x}}_0=\hat{{\mathbf{x}}}_0({\mathbf{x}}_t)},
$$

这在实践中可能带来显著计算开销。

代表性例子包括 [Yu et al., 2023](../../references.md#ref-yu2023freedom)、[Chung et al., 2022](../../references.md#ref-chung2022diffusion) 和 [Bansal et al., 2023](../../references.md#ref-bansal2023universal)。

上述构造规定了引导如何修改每一步采样。但仅凭这些构造，并不能确定将修改后的动力学跨所有噪声水平积分后得到的边缘分布。接下来澄清这一区别。

## 8.4.2 引导采样生成什么分布？ {#subsec-guided-marginal-path}

上述势函数视角在总体层面具有精确的固定时刻密度解释。具体来说，将学习到的无条件 score 估计 $\hat{{\mathbf{s}}}$ 替换为精确 score 后，正值势函数 ${\mathbf{h}}_t({\mathbf{x}};{\mathbf{c}})$ 与引导强度 $w_t\geq0$ 定义理想引导场

$$
\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}})
+
w_t\nabla_{{\mathbf{x}}}\log{\mathbf{h}}_t({\mathbf{x}};{\mathbf{c}}).
$$

对于每个固定 $t$，它都是以下密度的 score：

<span id="eq-guided-fixed-time-tilt"></span>

$$
\begin{aligned}
\widetilde p_t^{\mathrm{tilt}}({\mathbf{x}}|{\mathbf{c}})
=
\frac{1}{Z_t({\mathbf{c}})}
p_t({\mathbf{x}}){\mathbf{h}}_t({\mathbf{x}};{\mathbf{c}})^{w_t},
\qquad
Z_t({\mathbf{c}})
=
\int
p_t({\mathbf{x}}){\mathbf{h}}_t({\mathbf{x}};{\mathbf{c}})^{w_t}\,\mathop{}\!\mathrm{d}{\mathbf{x}},
\end{aligned}
\tag{8.4.3}
$$

前提是 $0<Z_t({\mathbf{c}})<\infty$。实践中使用的学习场，通过学习到的 score 估计 $\hat{{\mathbf{s}}}({\mathbf{x}}_t)$ 近似这一理想构造。

对于上述基于损失的构造，

$$
{\mathbf{h}}_t({\mathbf{x}}_t;{\mathbf{c}})
=
\exp\!\left[
-\tau\ell\bigl(\hat{{\mathbf{x}}}_0({\mathbf{x}}_t),{\mathbf{c}}\bigr)
\right],
$$

而 $\eta_t^{\mathrm{latent}}$ 起到 $w_t$ 的作用。

这一固定时刻恒等式并不能确定将引导动力学跨噪声水平积分后生成的分布。因此，对于一般损失与引导调度，从 $T$ 到 $0$ 求解引导后的 PF-ODE，一般不会产生服从 $\widetilde p_0^{\mathrm{tilt}}(\cdot|{\mathbf{c}})$ 的样本。

原因在于跨时间一致性：不同噪声水平的密度不能独立选择。要使 $\{\widetilde p_t^{\mathrm{tilt}}\}_t$ 构成同一前向扩散对应的边缘路径，它们还必须满足相应的 Fokker–Planck 方程：

$$
\partial_t \widetilde p_t^{\mathrm{tilt}}
+
\nabla_{{\mathbf{x}}} \cdot
\bigl(f(t){\mathbf{x}} \widetilde p_t^{\mathrm{tilt}}\bigr)
-
\frac{1}{2}g^2(t)
\Delta_{{\mathbf{x}}}\widetilde p_t^{\mathrm{tilt}}
=
0.
$$

这一条件成立时，相应 PF-ODE

$$
\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t}
=
f(t){\mathbf{x}}(t)
-
\frac{1}{2}g^2(t)
\nabla_{{\mathbf{x}}}\log
\widetilde p_t^{\mathrm{tilt}}\left({\mathbf{x}}(t)|{\mathbf{c}}\right)
$$

具有边缘路径 $\widetilde p_t^{\mathrm{tilt}}$，前提是从相应终端分布 $\widetilde p_T^{\mathrm{tilt}}(\cdot|{\mathbf{c}})$ 初始化。

可以直接检查相容条件。将[式 (8.4.3)](04-training-free-guidance.md#eq-guided-fixed-time-tilt)代入 Fokker–Planck 方程，并利用 $p_t$ 本身满足同一方程这一事实，可得

$$
\begin{aligned}
&
\partial_t \widetilde p_t^{\mathrm{tilt}}
+
\nabla_{{\mathbf{x}}} \cdot
\bigl(f(t){\mathbf{x}} \widetilde p_t^{\mathrm{tilt}}\bigr)
-
\frac{1}{2}g^2(t)
\Delta_{{\mathbf{x}}}\widetilde p_t^{\mathrm{tilt}}
\\
&=
\widetilde p_t^{\mathrm{tilt}}
\Bigg[
\partial_t
\log \bigl({\mathbf{h}}_t^{ w_t}\bigr)
+
f(t){\mathbf{x}}\cdot
\nabla_{{\mathbf{x}}}\log \bigl({\mathbf{h}}_t^{ w_t}\bigr)
-
g^2(t)
\nabla_{{\mathbf{x}}}\log p_t
\cdot
\nabla_{{\mathbf{x}}}\log \bigl({\mathbf{h}}_t^{ w_t}\bigr)
\\[-2pt]
&\hspace{3.4cm}
-
\frac{1}{2}g^2(t)
\left(
\Delta_{{\mathbf{x}}}\log \bigl({\mathbf{h}}_t^{ w_t}\bigr)
+
\left\|
\nabla_{{\mathbf{x}}}\log \bigl({\mathbf{h}}_t^{ w_t}\bigr)
\right\|^2
\right)
-
\frac{Z_t'}{Z_t}
\Bigg],
\end{aligned}
$$

为便于阅读，省略了自变量 $({\mathbf{x}};{\mathbf{c}})$。因此，只有方括号中的量为零时，倾斜密度族才是有效的边缘路径。一般的任务损失与引导调度不一定满足这一条件。所以，即使 score 计算及 ODE 积分完全精确，不匹配仍可能存在：它来自引导动力学本身，而非数值离散化。这是 [Lai et al., 2023](../../references.md#ref-lai2023fp) 讨论的更广泛 Fokker–Planck 一致性要求的一个实例。

两个简单情形可以澄清这一区别。当 $w_t\equiv0$ 时，倾斜消失，恢复原始无条件路径 $p_t$。对于精确分类器引导，在标准设定中，前向扩散扰动 ${\mathbf{x}}$ 而保持 ${\mathbf{c}}$ 不变，

$$
{\mathbf{h}}_t({\mathbf{x}};{\mathbf{c}})
=
p_t({\mathbf{c}}|{\mathbf{x}}),
\qquad
w_t\equiv1.
$$

Bayes 法则于是给出

$$
\widetilde p_t^{\mathrm{tilt}}({\mathbf{x}}|{\mathbf{c}})
=
p_t({\mathbf{x}}|{\mathbf{c}}),
$$

因此，倾斜密度族恰好就是条件边缘路径。这解释了 oracle 分类器似然的引导权重 $1$ 为何具有特殊作用。

对于一般引导强度，例如

$$
p_t({\mathbf{x}}) p_t({\mathbf{c}}|{\mathbf{x}})^\omega,
$$

只要可以归一化，这个表达式仍定义了有效的固定时刻倾斜密度，但通常不能保证它描述引导采样器的边缘分布。某些特殊选择可能满足所需相容条件，因此仅有 $\omega\neq1$ 并不意味着不一致。

当目标是更忠实地遵循给定倾斜分布序列时，Feynman-Kac Correctors 等基于校正的方法（[Skreta et al., 2025](../../references.md#ref-skreta2025feynman)）会显式处理这种跨时间不匹配。

## 8.4.3 用于逆问题的免训练方法示例 {#subsec-inverse-problem}

[第 8.4.1 节](04-training-free-guidance.md#subsec-tfg)介绍的原理在逆问题中有重要应用。下面先概述背景，再给出若干具体例子，说明如何使用预训练扩散模型在推断时求解逆问题。

**逆问题背景。** 令 $\mathcal A$ 表示已知的、可能非线性的前向算子，并假设

<span id="eq-inverse-problem"></span>

$$
\begin{aligned}
{\mathbf{y}}
=
\mathcal A({\mathbf{x}}_0)
+
\sigma_{{\mathbf{y}}}{\mathbf{z}},
\qquad
{\mathbf{z}}\sim\mathcal N(\boldsymbol 0,{\mathbf{I}}).
\end{aligned}
\tag{8.4.4}
$$

Bayes 目标是刻画后验 $p_0({\mathbf{x}}_0|{\mathbf{y}})$，它将测量模型与关于合理干净信号的先验知识结合起来。盲逆问题进一步将 $\mathcal A$ 的未知参数视为待推断的额外量。

经典逆问题方法通常在变分或 Bayes 表述中，将数据保真项与解析先验或人工设计的先验结合。监督式复原方法则可以使用配对数据，直接学习从测量到干净信号的映射。预训练扩散模型提供了第三条途径：学习到的干净数据先验可以在不同测量算子之间复用，而测量信息只在推断时引入。

**预训练扩散模型作为逆问题求解器。** 对于上述测量模型，Bayes 法则给出

<span id="eq-bayes-rule"></span>

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t|{\mathbf{y}})
=
\underbrace{\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t)}_{\text{data score}}
+
\underbrace{\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{y}}|{\mathbf{x}}_t)}_{\text{measurement guidance}}.
\end{aligned}
\tag{8.4.5}
$$

数据 score 可以由预训练扩散模型提供，而测量项通常没有可用的闭式形式。因此，免训练逆问题求解器会在采样时引入测量信息，但不必都采用同一种数学形式。

可以区分两种常见策略。一些方法通过修改或投影中间样本，直接强制测量一致性。另一些方法则近似似然 score $\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{y}}|{\mathbf{x}}_t)$，并通过[式 (8.4.5)](04-training-free-guidance.md#eq-bayes-rule)将其加到数据 score 上。

**ILVR（[Choi et al., 2021](../../references.md#ref-choi2021ilvr)）。** ILVR 是第一种策略的简单例子。令 $\phi_N$ 表示其线性低通算子。每个逆向步骤中，先采样一个无条件候选样本

$$
{\mathbf{x}}_{t-1}'
\sim
p_{\boldsymbol\theta}({\mathbf{x}}_{t-1}'|{\mathbf{x}}_t)
$$

并将参考图像加噪到相同水平，

$$
{\mathbf{y}}_{t-1}
\sim
p_{t-1}({\mathbf{y}}_{t-1}|{\mathbf{y}}).
$$

然后按如下方式改进候选样本

$$
{\mathbf{x}}_{t-1}
=
\phi_N({\mathbf{y}}_{t-1})
+
({\mathbf{I}}-\phi_N){\mathbf{x}}_{t-1}'.
$$

因此，给定的低频分量来自加噪参考图像，而互补分量来自无条件扩散候选样本。这是样本空间中的一致性修正，并非对 $\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{y}}|{\mathbf{x}}_t)$ 的显式近似。

**Diffusion Posterior Sampling（DPS）（[Chung et al., 2022](../../references.md#ref-chung2022diffusion)）。** 对于已知的非线性前向算子 $\mathcal A$ 及加性 Gaussian 测量噪声 $\sigma_{{\mathbf{y}}}>0$，*Diffusion Posterior Sampling*（DPS）采用如下近似

<span id="eq-dps-score"></span>

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{y}}|{\mathbf{x}}_t)
\approx
\nabla_{{\mathbf{x}}_t}
\log p\bigl({\mathbf{y}}|X_0=\hat{\mathbf{x}}_0({\mathbf{x}}_t)\bigr),
\end{aligned}
\tag{8.4.6}
$$

其中，$\hat{\mathbf{x}}_0({\mathbf{x}}_t):={\mathbb{E}}[{\mathbf{x}}_0|{\mathbf{x}}_t]$ 是给定 ${\mathbf{x}}_t$ 时干净样本的后验均值，通常使用 Tweedie 公式（[式 (3.3.6)](../03-score-based-perspective/03-denoising-score-matching.md#eq-tweedie)）从预训练扩散模型中估计。

这一代入近似用测量似然在后验均值处的值，替代测量似然的后验平均：

$$
\begin{aligned}
p_t({\mathbf{y}}|{\mathbf{x}}_t)
  &= \int p_t({\mathbf{y}}|{\mathbf{x}}_t, {\mathbf{x}}_0)  p({\mathbf{x}}_0|{\mathbf{x}}_t)  \mathrm{d}{\mathbf{x}}_0 \\
  &= \int p_t({\mathbf{y}}|{\mathbf{x}}_0)  p({\mathbf{x}}_0|{\mathbf{x}}_t)  \mathrm{d}{\mathbf{x}}_0
   \approx  p_t\bigl({\mathbf{y}}|X_0=\hat{\mathbf{x}}_0({\mathbf{x}}_t)\bigr),
\end{aligned}
$$

其中使用了以下事实：${\mathbf{y}}$ 仅依赖 ${\mathbf{x}}_0$（不依赖 ${\mathbf{x}}_t$），条件是给定 ${\mathbf{x}}_0$；这一近似在后验 $p({\mathbf{x}}_0|{\mathbf{x}}_t)$ 紧密集中于其均值附近的假设下成立。

由于

$$
p_t\bigl({\mathbf{y}}|X_0 = \hat{\mathbf{x}}_0({\mathbf{x}}_t)\bigr)
  = \mathcal{N}\bigl({\mathbf{y}};  \mathcal{A}(\hat{\mathbf{x}}_0({\mathbf{x}}_t)),  \sigma_{{\mathbf{y}}}^2 {\mathbf{I}}\bigr),
$$

可计算

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{y}}|{\mathbf{x}}_t)
  &\approx \nabla_{{\mathbf{x}}_t} \log \mathcal{N}\bigl({\mathbf{y}};  \mathcal{A}(\hat{\mathbf{x}}_0),  \sigma_{{\mathbf{y}}}^2 {\mathbf{I}}\bigr) \\
  &= -\frac{1}{2\sigma_{{\mathbf{y}}}^2}  \nabla_{{\mathbf{x}}_t} \bigl\| {\mathbf{y}} - \mathcal{A}(\hat{\mathbf{x}}_0) \bigr\|^2 \quad  \\
  &= \frac{1}{\sigma_{{\mathbf{y}}}^2}  \left[ \mathcal{J}_{\mathcal{A}}\big(\hat{\mathbf{x}}_0({\mathbf{x}}_t)\big) \cdot \nabla_{{\mathbf{x}}_t} \hat{\mathbf{x}}_0({\mathbf{x}}_t) \right]^\top \bigl( {\mathbf{y}} - \mathcal{A}\big(\hat{\mathbf{x}}_0({\mathbf{x}}_t)\big) \bigr),
\end{aligned}
$$

其中，$\mathcal{J}_{\mathcal{A}}\big(\hat{\mathbf{x}}_0({\mathbf{x}}_t)\big) := \nabla_{{\mathbf{x}}_0} \mathcal{A}({\mathbf{x}}) \big|_{{\mathbf{x}} = \hat{\mathbf{x}}_0({\mathbf{x}}_t)}$ 表示前向算子相对于其输入的 Jacobian 矩阵。该公式使梯度沿 score 近似流程传播，反映测量似然如何随带噪样本 ${\mathbf{x}}_t$ 的扰动而变化。

对于线性逆问题，它进一步简化为：

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{y}} | {\mathbf{x}}_t) \approx \bbox[2px,#d6e5c9]{\color{black}{\frac{1}{\sigma_{{\mathbf{y}}}^2}}} \,\,    \bbox[2px,#e7cfea]{\color{black}{\left[ {\mathbf{A}} \cdot \nabla_{{\mathbf{x}}_t} \hat{\mathbf{x}}_0({\mathbf{x}}_t) \right]^\top}} \left(\bbox[2px,#d5dcf2]{\color{black}{{\mathbf{y}} - {\mathbf{A}}(\hat{\mathbf{x}}_0({\mathbf{x}}_t))}}\right).
\end{aligned}
$$

大量工作通过提出对 $\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{y}} | {\mathbf{x}}_t)$ 的不同近似，探索基于扩散的逆问题求解器。全面介绍参见 [Daras et al. (2024)](../../references.md#ref-daras2024survey) 的综述。

**相关的监督式迭代复原视角及其与 PF-ODE 的联系。** 上述讨论通过将预训练扩散先验与[式 (8.4.5)](04-training-free-guidance.md#eq-bayes-rule)中的 Bayes 分解结合来求解逆问题。具体而言，这些方法同时使用先验分布的 score 和测量模型，引导重建过程。同时，它们也体现了一个更广泛的思想：通过许多小步更新，逐步改进受扰动输入，而非一步恢复干净信号。监督式图像复原中也出现了类似的整体策略，但出发点不同。这些方法从配对的受扰动样本与干净样本中直接学习改进过程，而不将预训练 score 模型用作先验。

代表性例子是 InDI（[Delbracio and Milanfar, 2023](../../references.md#ref-delbracio2023inversion)）。与上述基于 Bayes 法则的方法不同，InDI 从配对数据 $({\mathbf{y}},{\mathbf{x}}_0)$ 出发，其中 ${\mathbf{y}}$ 是受扰动观测，${\mathbf{x}}_0$ 是对应的干净目标。它通常不假设从 ${\mathbf{x}}_0$ 到 ${\mathbf{y}}$ 的扰动具有已知解析形式，也不假设扰动是 Gaussian 的。

基本思想是通过插值路径连接两个端点

$$
{\mathbf{x}}_t=(1-t){\mathbf{x}}_0+t{\mathbf{y}},\qquad t\in[0,1].
$$

这条路径应仅理解为从干净目标到受扰动观测的一条方便轨迹。它一般不是 Gaussian 加噪过程。沿这条路径，对于任意 $0\le s\le t$，有精确关系

$$
{\mathbf{x}}_s=\Bigl(1-\frac{s}{t}\Bigr){\mathbf{x}}_0+\frac{s}{t}{\mathbf{x}}_t.
$$

给定 ${\mathbf{x}}_t$ 取条件期望，得到

$$
\mathbb E[{\mathbf{x}}_s|{\mathbf{x}}_t]
=
\Bigl(1-\frac{s}{t}\Bigr)\mathbb E[{\mathbf{x}}_0|{\mathbf{x}}_t]
+
\frac{s}{t}{\mathbf{x}}_t.
$$

这说明，为从当前状态 ${\mathbf{x}}_t$ 移到退化程度略低的状态，只需根据 ${\mathbf{x}}_t$ 预测干净目标。

基于这一思想，通过最小化回归目标，从配对样本中训练以时间为条件的预测器 ${\mathbf{F}}_{{\boldsymbol{\phi}}}({\mathbf{x}}_t,t)$：

$$
\min_{{\boldsymbol{\phi}}}\;
\mathbb E_{({\mathbf{x}}_0,{\mathbf{y}})\sim p({\mathbf{x}}_0,{\mathbf{y}}),  t\sim p(t)}
\Bigl\|
{\mathbf{F}}_{{\boldsymbol{\phi}}}\bigl((1-t){\mathbf{x}}_0+t{\mathbf{y}}, t\bigr)-{\mathbf{x}}_0
\Bigr\|_2^2,
$$

其中，$p({\mathbf{x}}_0,{\mathbf{y}})$ 表示数据对的联合分布，每对包含干净目标 ${\mathbf{x}}_0$ 及其受扰动观测 ${\mathbf{y}}$；$p(t)$ 指定训练中如何采样插值时间。也就是说，训练 ${\mathbf{F}}_{{\boldsymbol{\phi}}}$，使其根据插值路径上的中间退化状态预测干净目标。对于这一平方误差损失，Bayes 最优预测器是条件期望

$$
{\mathbf{F}}^*({\mathbf{x}}_t,t)=\mathbb E[{\mathbf{x}}_0|{\mathbf{x}}_t].
$$

因此，如果模型类具有足够表达能力且优化成功，学习到的预测器就近似于与中间状态 ${\mathbf{x}}_t$（时刻 $t$）对应的平均干净目标。在这一意义上，该训练目标与扩散模型训练的思想相似：两者都学习以时间为条件、从中间受扰动状态到更干净目标的回归。区别在于，扩散模型通过给定前向扰动过程合成训练对，而这里的端点对 $({\mathbf{y}},{\mathbf{x}}_0)$ 直接来自监督式复原数据。

推断时，从 $\hat{{\mathbf{x}}}_1={\mathbf{y}}$ 出发，反复按下式改进估计

$$
\hat{{\mathbf{x}}}_{t-\Delta t}
=
\frac{\Delta t}{t}{\mathbf{F}}_{{\boldsymbol{\phi}}}(\hat{{\mathbf{x}}}_t,t)
+
\Bigl(1-\frac{\Delta t}{t}\Bigr)\hat{{\mathbf{x}}}_t.
$$

因此，最终重建通过逐步改进产生，而非单次直接预测。在连续时间极限下，这得到 InDI 的残差流 ODE

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}_t}{\mathop{}\!\mathrm{d} t}
=
\frac{{\mathbf{x}}_t-{\mathbf{F}}_{{\boldsymbol{\phi}}}({\mathbf{x}}_t,t)}{t}.
$$

现在进一步考虑 Gaussian 去噪这一特殊情形，假设扰动具有已知解析形式

$$
{\mathbf{y}}={\mathbf{x}}_0+\sigma{\boldsymbol{\epsilon}},
\qquad {\boldsymbol{\epsilon}}\sim\mathcal N(\boldsymbol 0,{\mathbf{I}}),
$$

其中噪声水平 $\sigma$ 已知。在这一特殊设定下，InDI 的残差流 ODE 恰好化为概率流 ODE。确实，插值路径变为

$$
{\mathbf{x}}_t=(1-t){\mathbf{x}}_0+t{\mathbf{y}}
=
{\mathbf{x}}_0+t\sigma{\boldsymbol{\epsilon}},
$$

这正是标准扩散形式

$$
{\mathbf{x}}_t=\alpha_t{\mathbf{x}}_0+\sigma_t{\boldsymbol{\epsilon}},
\qquad\text{with}\qquad
\alpha_t=1,\;\sigma_t=t\sigma.
$$

因此，预测器 ${\mathbf{F}}_{{\boldsymbol{\phi}}}({\mathbf{x}}_t,t)\approx \mathbb E[{\mathbf{x}}_0\mid {\mathbf{x}}_t]$ 此时学习对受 Gaussian 扰动的中间状态 ${\mathbf{x}}_t$ 进行去噪。在这种情况下，残差流 ODE 变为

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}_t}{\mathop{}\!\mathrm{d} t}
=
\frac{{\mathbf{x}}_t-\mathbb E[{\mathbf{x}}_0\mid {\mathbf{x}}_t]}{t}
=
-\dot{\sigma}_t\sigma_t\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t),
$$

最后一个等号由 $\sigma_t=t\sigma$ 结合 Tweedie 公式得到，

$$
\mathbb E[{\mathbf{x}}_0\mid {\mathbf{x}}_t]
=
{\mathbf{x}}_t+\sigma_t^2\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t).
$$

这正是该 Gaussian 去噪路径的 PF-ODE，可写成去噪器形式或 score 形式。因此，在 Gaussian 去噪这一特殊情形下，InDI 提供了扩散建模的另一种解释：可以将其视为在受 Gaussian 扰动的数据上进行监督式去噪器学习，而 PF-ODE 则是逐步改进的连续时间极限。

[^1]: 为简单起见，这里采用 score 和 ${\mathbf{x}}$ 预测参数化；${\boldsymbol{\epsilon}}$ 预测等其他参数化可以类似处理。
