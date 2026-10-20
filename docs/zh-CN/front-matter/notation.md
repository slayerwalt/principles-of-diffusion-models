# 符号说明 {#notations}

## 数与数组 {#notation-numbers-and-arrays}

| 符号 | 含义 |
| --- | --- |
| $a$ | 标量。 |
| ${\mathbf{a}}$ | 列向量（例如，${\mathbf{a}}\in\mathbb{R}^D$）。 |
| ${\mathbf{A}}$ | 矩阵（例如，${\mathbf{A}}\in\mathbb{R}^{m\times n}$）。 |
| ${\mathbf{A}}^\top$ | ${\mathbf{A}}$ 的转置。 |
| $\operatorname{Tr}({\mathbf{A}})$ | ${\mathbf{A}}$ 的迹。 |
| ${\mathbf{I}}_D$ | 大小为 $D\times D$ 的单位矩阵。 |
| ${\mathbf{I}}$ | 单位矩阵，维度由上下文确定。 |
| $\mathrm{diag}({\mathbf{a}})$ | 以 ${\mathbf{a}}$ 的各分量为对角元素的对角矩阵。 |
| $\boldsymbol{\phi},\,\boldsymbol{\theta}$ | 可学习的神经网络参数。 |
| $\boldsymbol{\phi}^{\times},\,\boldsymbol{\theta}^{\times}$ | 训练后的参数（推断时固定不变）。 |
| $\boldsymbol{\phi}^{*},\,\boldsymbol{\theta}^{*}$ | 优化问题的最优参数。 |

## 微积分 {#notation-calculus}

| 符号 | 含义 |
| --- | --- |
| $\dfrac{\partial {\mathbf{y}}}{\partial {\mathbf{x}}}$ | ${\mathbf{y}}$ 关于 ${\mathbf{x}}$ 的偏导数（逐分量计算）。 |
| $\dfrac{\mathop{}\!\mathrm{d} {\mathbf{y}}}{\mathop{}\!\mathrm{d} {\mathbf{x}}}$ 或 $\mathrm{D}{\mathbf{y}}({\mathbf{x}})$ | ${\mathbf{y}}$ 关于 ${\mathbf{x}}$ 的全导数（Fréchet 导数）。 |
| $\nabla_{\mathbf{x}} y$ | 标量函数 $y:\mathbb{R}^D \to\mathbb{R}$ 的梯度，是 $\mathbb{R}^D$ 中的列向量。 |
| $\dfrac{\partial {\mathbf{F}}}{\partial {\mathbf{x}}}$ 或 $\nabla_{\mathbf{x}} {\mathbf{F}}$ | ${\mathbf{F}}:\mathbb{R}^n \to\mathbb{R}^m$ 的 Jacobian 矩阵，大小为 $m\times n$。 |
| $\nabla\cdot{\mathbf{y}}$ | 向量场 ${\mathbf{y}}:\mathbb{R}^D \to\mathbb{R}^D$ 的散度，是一个标量。 |
| $\nabla^2_{\mathbf{x}} f({\mathbf{x}})$ 或 ${\mathbf{H}}(f)({\mathbf{x}})$ | $f:\mathbb{R}^D \to\mathbb{R}$ 的 Hessian 矩阵，大小为 $D\times D$。 |
| $\displaystyle\int f({\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}$ | $f$ 在 ${\mathbf{x}}$ 的定义域上的积分。 |

## 概率与信息论 {#notation-probability-and-information-theory}

| 符号 | 含义 |
| --- | --- |
| $p({\mathbf{x}})$ | 连续向量 ${\mathbf{x}}$ 的密度／分布。 |
| $\Pr({\mathbf{x}} = {\mathbf{a}})$，或非正式地写为 $\Pr({\mathbf{x}})$ | 离散情形下的点概率。 |
| $\Pr({\mathbf{x}} = {\mathbf{a}} \vert {\mathbf{y}} = {\mathbf{b}})$，或非正式地写为 $\Pr({\mathbf{x}} \vert {\mathbf{y}})$ | 离散情形下的条件点概率。 |
| $p_{\mathrm{data}}$ | 数据分布。 |
| $p_{\mathrm{prior}}$ | 先验分布（例如，标准正态分布）。 |
| $p_{\mathrm{src}}$ | 源分布。 |
| $p_{\mathrm{tgt}}$ | 目标分布。 |
| ${\mathbf{a}} \sim p$ | 随机向量 ${\mathbf{a}}$ 服从分布 $p$。 |
| ${\mathbb{E}}_{{\mathbf{x}}\sim p}\big[{\mathbf{f}}({\mathbf{x}})\big]$ | ${\mathbf{f}}({\mathbf{x}})$ 在 $p({\mathbf{x}})$ 下的期望。 |
| ${\mathbb{E}}\big[{\mathbf{f}}({\mathbf{x}})\,\vert \,{\mathbf{z}}\big]$，或 ${\mathbb{E}}_{{\mathbf{x}}\sim p(\cdot\vert {\mathbf{z}})}\big[{\mathbf{f}}({\mathbf{x}})\big]$ | ${\mathbf{f}}({\mathbf{x}})$ 在给定 ${\mathbf{z}}$ 时的条件期望，其中 ${\mathbf{x}}$ 服从分布 $p(\cdot\vert {\mathbf{z}})$。 |
| $\mathrm{Var}\big({\mathbf{f}}({\mathbf{x}})\big)$ | $p({\mathbf{x}})$ 下的方差。 |
| $\mathrm{Cov}\big({\mathbf{f}}({\mathbf{x}}),{\mathbf{g}}({\mathbf{x}})\big)$ | $p({\mathbf{x}})$ 下的协方差。 |
| $\mathcal{D}_{\mathrm{KL}}\left(p\,\Vert\, q\right)$ | 从 $q$ 到 $p$ 的 KL散度。 |
| $\boldsymbol{\epsilon}\sim\mathcal{N}(\boldsymbol{0},{\mathbf{I}})$ | 标准正态样本。 |
| $\mathcal{N}({\mathbf{x}};{\boldsymbol{\mu}},{\boldsymbol{\Sigma}})$ | 关于 ${\mathbf{x}}$ 的 Gaussian 分布，均值为 ${\boldsymbol{\mu}}$，协方差为 ${\boldsymbol{\Sigma}}$。 |

## 符号约定 {#notational-conventions}

这里汇总本书通用的若干约定。

### I. 随机变量及其实现值 {#i-random-variables-and-their-realizations}

我们用同一个符号表示随机向量及其实现值。这一约定在深度学习与生成建模中很常见，能够使符号保持简洁。具体含义由上下文确定。

对于密度，$p({\mathbf{x}})$ 表示分布的函数形式，而非在某个特定样本处的取值。如果要表示在某个特定点处的取值，我们会明确说明。条件密度的含义由上下文确定：在 $p({\mathbf{x}}|{\mathbf{y}})$ 中，固定 ${\mathbf{y}}$ 后得到关于 ${\mathbf{x}}$ 的密度；固定 ${\mathbf{x}}$ 后则得到关于 ${\mathbf{y}}$ 的函数。

### II. 条件期望 {#ii-conditional-expectations}

表达式 ${\mathbb{E}}[{\mathbf{f}}({\mathbf{x}})|{\mathbf{z}}]$ 表示关于 ${\mathbf{z}}$ 的函数，给出 ${\mathbf{f}}({\mathbf{x}})$ 在给定 ${\mathbf{z}}$ 时的期望值。当条件是某个具体的实现值时，我们写作 ${\mathbb{E}}[{\mathbf{f}}({\mathbf{x}})|{\mathbf{Z}} = {\mathbf{z}}]$。等价地，它可以表示为对条件分布的积分：

$$
{\mathbb{E}}_{{\mathbf{x}} \sim p(\cdot|{\mathbf{z}})}[{\mathbf{f}}({\mathbf{x}})]
= \int {\mathbf{f}}({\mathbf{x}})\, p({\mathbf{x}}|{\mathbf{z}})\, \mathop{}\!\mathrm{d}{\mathbf{x}}.
$$

这一区分明确了 ${\mathbf{z}}$ 是用于定义函数 ${\mathbf{z}} \mapsto {\mathbb{E}}[{\mathbf{f}}({\mathbf{x}})|{\mathbf{z}}]$ 的变量，还是用于计算该函数取值的固定值。

### III. 隐含的时间索引 {#iii-implicit-time-indices}

本书中的许多对象都以时间为索引，显式写出每个时间变量会使公式繁杂。当相关时间索引可以从上下文明确得知时，我们会省略它们。例如，

$$
{\mathbb{E}}[{\mathbf{x}}_s | {\mathbf{x}}_t]
\quad\text{is shorthand for}\quad
{\mathbb{E}}[{\mathbf{x}}_s | {\mathbf{x}}_t,\, s,\, t],
$$

表示 ${\mathbf{x}}_s$ 在时刻 $s$ 的平均值，条件是给定 ${\mathbf{x}}_t$ 在时刻 $t$ 的取值，其中 $s$ 和 $t$ 可从上下文理解。同样，$p({\mathbf{x}}_s | {\mathbf{x}}_t)$ 表示 ${\mathbf{x}}_s$ 在给定 ${\mathbf{x}}_t$ 时的条件分布，两个时间索引均隐含于其中。

### IV. Gaussian 扰动与依分布相等 {#iv-gaussian-perturbation-and-equality-in-distribution}

对于固定的索引 $t$，我们经常使用 Gaussian 扰动核

$$
p_t({\mathbf{x}}_t|{\mathbf{x}}_0)
= \mathcal{N}\big({\mathbf{x}}_t;\, \alpha_t {\mathbf{x}}_0,\, \sigma_t^2 {\mathbf{I}}\big),
$$

其中 ${\mathbf{x}}_0 \sim p_{\mathrm{data}}$，$\alpha_t$ 和 $\sigma_t$ 是确定性标量。等价地，我们将其写为*依分布相等*的等式

$$
{\mathbf{x}}_t \stackrel{d}{=} \alpha_t {\mathbf{x}}_0 + \sigma_t {\boldsymbol{\epsilon}},
\qquad
{\boldsymbol{\epsilon}} \sim \mathcal{N}(\boldsymbol{0}, {\mathbf{I}}),
$$

其中 ${\boldsymbol{\epsilon}}$ 与 ${\mathbf{x}}_0$ 独立。符号“$\stackrel{d}{=}$”表示两侧具有相同的概率律，因此

$$
{\mathbb{E}}[\phi({\mathbf{x}}_t)] = {\mathbb{E}}[\phi(\alpha_t {\mathbf{x}}_0 + \sigma_t {\boldsymbol{\epsilon}})]
$$

对任意测试函数 $\phi$ 都成立。为简洁起见，我们将直接写作 ${\mathbf{x}}_t = \alpha_t {\mathbf{x}}_0 + \sigma_t {\boldsymbol{\epsilon}}$，根据上下文将其理解为依分布相等，或一次样本实现；本书始终采用这一简写。
