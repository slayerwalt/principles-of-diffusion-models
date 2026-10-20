# 4.6 （选读）通过边缘化与 Bayes 法则推导 Fokker–Planck 方程及逆时间 SDE {#eq-heuristic-fpe-sde}

本节从概率视角考察 Fokker–Planck 方程与逆时间 SDE 的结构。借助边缘化技巧与 Bayes 法则等基本工具，我们阐明随机过程的统计表述与相应微分方程之间的联系。

这里的“推导”并非数学上的严格证明，而是旨在传达底层联系的启发式论证。

## 4.6.1 从转移核的边缘化得到 Fokker–Planck 方程 {#subsec-fpe-reason}

给定[式 (4.1.2)](01-score-sde-principles.md#eq-transition-forward-f-g)中的前向转移概率

$$
\begin{aligned}
p(\mathbf{x}_{t+\Delta t} | \mathbf{x}_t) = \mathcal{N}\left(\mathbf{x}_{t+\Delta t}; \mathbf{x}_t + \mathbf{f}(\mathbf{x}_t, t)\Delta t, g^2(t)\Delta t \mathbf{I}\right),
\end{aligned}
$$

以及边缘分布

$$
\begin{aligned}
p_t(\mathbf{x}_t), \quad p_{t+\Delta t}(\mathbf{x}_{t+\Delta t}),
\end{aligned}
$$

我们希望推导支配边缘分布 $p_t$ 随时间演化的 Fokker–Planck 方程。

**变量替换。** 根据 Markov 性质，时刻 $t + \Delta t$ 的边缘分布可表示为对前一状态 $\mathbf{x}_t$ 的积分（即 Chapman–Kolmogorov 方程）：

$$
\begin{aligned}
p_{t+\Delta t}({\mathbf{x}})
= \int \mathcal N \big({\mathbf{x}}; {\mathbf{y}}+{\mathbf{f}}({\mathbf{y}},t)\Delta t, g^2(t)\Delta t {\mathbf{I}}\big) p_t({\mathbf{y}}) \mathop{}\!\mathrm{d}{\mathbf{y}}.
\end{aligned}
$$

引入新变量

$$
{\mathbf{u}}  :=  {\mathbf{y}}+{\mathbf{f}}({\mathbf{y}},t)\Delta t,
$$

使 Gaussian 分布以 ${\mathbf{u}}$ 为中心。对于较小的 $\Delta t$，这一映射可逆，且

$$
{\mathbf{y}}  =  {\mathbf{u}}-{\mathbf{f}}({\mathbf{u}},t)\Delta t+\mathcal O(\Delta t^2),
\quad
\Bigg|\det\frac{\partial {\mathbf{y}}}{\partial {\mathbf{u}}}\Bigg|
 = 1-(\nabla_{\mathbf{u}} \cdot{\mathbf{f}})({\mathbf{u}},t) \Delta t+\mathcal O(\Delta t^2).
$$

因此，变量替换公式给出：

$$
\begin{aligned}
p_{t+\Delta t}({\mathbf{x}})
&= \int \mathcal N\left({\mathbf{x}};{\mathbf{u}},g^2(t)\Delta t{\mathbf{I}}\right) \cdot
\\\quad\Big[p_t({\mathbf{u}})&-\Delta t {\mathbf{f}}({\mathbf{u}},t) \cdot \nabla_{\mathbf{u}} p_t({\mathbf{u}})
-\Delta t (\nabla_{\mathbf{u}} \cdot{\mathbf{f}})({\mathbf{u}},t) p_t({\mathbf{u}})\Big] \mathop{}\!\mathrm{d}{\mathbf{u}}
+\mathcal O(\Delta t^2),
\end{aligned}
$$

**Taylor 展开。** 对任意光滑函数 $\phi:{\mathbb{R}}^D\to{\mathbb{R}}$ 与尺度 $\sigma>0$，若 ${\mathbf{z}}\sim\mathcal N(\boldsymbol{0},{\mathbf{I}})$，则以下近似成立（称为 *Taylor–Gaussian 平滑公式*）：

$$
\int \mathcal N({\mathbf{x}};{\mathbf{u}},\sigma^2{\mathbf{I}}) \phi({\mathbf{u}}) \mathop{}\!\mathrm{d}{\mathbf{u}}
={\mathbb{E}} \left[\phi({\mathbf{x}}+\sigma{\mathbf{z}})\right]
=\phi({\mathbf{x}})+\frac{\sigma^2}{2} \Delta_{\mathbf{x}}\phi({\mathbf{x}})+\mathcal O(\sigma^4).
$$

这是因为以下 Taylor 展开成立：

$$
\phi({\mathbf{x}}+\sigma{\mathbf{z}})=\phi({\mathbf{x}})+\sigma\nabla_{\mathbf{x}}\phi({\mathbf{x}}) \cdot {\mathbf{z}}+\frac{\sigma^2}{2}{\mathbf{z}}^\top\nabla_{\mathbf{x}}^2\phi({\mathbf{x}}){\mathbf{z}}+\mathcal O(\sigma^3)
$$

且 ${\mathbb{E}}[{\mathbf{z}}]=\boldsymbol{0}$、${\mathbb{E}}[{\mathbf{z}}{\mathbf{z}}^\top]={\mathbf{I}}$。

将其应用于 $\phi=p_t$、$\phi={\mathbf{f}} \cdot \nabla_{\mathbf{u}} p_t$ 与 $\phi=(\nabla_{\mathbf{u}} \cdot{\mathbf{f}}) p_t$，并利用 $\sigma^2=g^2(t)\Delta t$，可得

$$
\begin{aligned}
&p_{t+\Delta t}({\mathbf{x}}) - p_t({\mathbf{x}})
\\=& -\Delta t {\mathbf{f}}({\mathbf{x}},t) \cdot \nabla_{\mathbf{x}} p_t({\mathbf{x}})
-\Delta t (\nabla_{\mathbf{x}} \cdot{\mathbf{f}})({\mathbf{x}},t) p_t({\mathbf{x}})
+\frac{g^2(t)}{2}\Delta t \Delta_{\mathbf{x}} p_t({\mathbf{x}})
+\mathcal O(\Delta t^2)
\\=&  - \Delta t \nabla_{\mathbf{x}} \cdot \big({\mathbf{f}}({\mathbf{x}},t) p_t({\mathbf{x}})\big)
 + \frac{g^2(t)}{2} \Delta t \Delta_{\mathbf{x}} p_t({\mathbf{x}})
 + \mathcal O(\Delta t^2).
\end{aligned}
$$

除以 $\Delta t$，再令 $\Delta t\to0$，即得到 Fokker–Planck 方程。

[第 C.1.4 节](../../appendices/c-ito-calculus/01-ito-formula.md#subsec-rigorous-proof-fpe)给出基于 Itô 微积分的推导，以补充上述离散时间视角。

## 4.6.2 逆时间 SDE 为什么具有这一形式？ {#subsec-reverse-sde-reason}

逆时间 SDE 的严格推导具有较强技术性，需要详细研究 Fokker–Planck 方程的性质。然而，通过 Bayes 定理可以直观理解逆时间 SDE 的形式。这里给出启发式推导，帮助理解[式 (4.1.6)](01-score-sde-principles.md#eq-sde-backward)为什么具有包含 score 函数的形式[^1]。

**使用 Bayes 法则进行逆转。** 我们首先考虑离散时间情形，以确定逆时间转移核：

$$
p({\mathbf{x}}_t | {\mathbf{x}}_{t+\Delta t}),
$$

然后令 $\Delta t \rightarrow 0$，得到连续时间形式。利用 Bayes 定理，写为：

<span id="eq-reverse-sde-bayes-discrete"></span>

$$
\begin{aligned}
p({\mathbf{x}}_t | {\mathbf{x}}_{t+\Delta t})
    &= p({\mathbf{x}}_{t+\Delta t} | {\mathbf{x}}_t) \frac{p_t({\mathbf{x}}_t)}{p_{t+\Delta t}({\mathbf{x}}_{t+\Delta t})}  \\
    &= p({\mathbf{x}}_{t+\Delta t} | {\mathbf{x}}_t) \exp\left(\log p_t({\mathbf{x}}_t) - \log p_{t+\Delta t}({\mathbf{x}}_{t+\Delta t})\right).
\end{aligned}
\tag{4.6.1}
$$

假设前向转移核具有[式 (4.1.2)](01-score-sde-principles.md#eq-transition-forward-f-g)的形式：

$$
p({\mathbf{x}}_{t+\Delta t} | {\mathbf{x}}_t) = \mathcal{N}\left({\mathbf{x}}_{t+\Delta t}; {\mathbf{x}}_t + {\mathbf{f}}({\mathbf{x}}_t, t)\Delta t, g^2(t)\Delta t \mathbf{I}\right)
$$

**Taylor 展开。** 为处理指数项，应用一阶 Taylor 展开。关键是在空间与时间上同时围绕点 $({\mathbf{x}}_t, t)$ 展开：

$$
\begin{aligned}
\log p_{t+\Delta t}({\mathbf{x}}_{t+\Delta t}) &= \log p_t({\mathbf{x}}_t) + \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t) \cdot ({\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t)  \\
    &\quad + \frac{\partial \log p_t({\mathbf{x}}_t)}{\partial t} \Delta t + \mathcal{O}(\|{\mathbf{h}}\|_2^2)
\end{aligned}
$$

其中 ${\mathbf{h}} := ({\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t, \Delta t)$。因此：

<span id="eq-log-difference"></span>

$$
\begin{aligned}
\log p_t({\mathbf{x}}_t) - \log p_{t+\Delta t}({\mathbf{x}}_{t+\Delta t}) &= -\nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t) \cdot ({\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t)  \\
    &\quad - \frac{\partial \log p_t({\mathbf{x}}_t)}{\partial t} \Delta t + \mathcal{O}(\|{\mathbf{h}}\|_2^2)
\end{aligned}
\tag{4.6.2}
$$

对于扩散过程，${\mathbf{x}}_{t+\Delta t}-{\mathbf{x}}_t=\mathcal O_p(\sqrt{\Delta t})$，因此二阶空间 Taylor 项通常为 $\mathcal O_p(\Delta t)$ 阶。相应地，下述计算应理解为主导阶的启发式推导。

**代入逆向转移。** 将[式 (4.1.2)](01-score-sde-principles.md#eq-transition-forward-f-g)与[式 (4.6.2)](06-fokker-planck-and-reverse-sde.md#eq-log-difference)代入[式 (4.6.1)](06-fokker-planck-and-reverse-sde.md#eq-reverse-sde-bayes-discrete)：

$$
\begin{aligned}
&p({\mathbf{x}}_t | {\mathbf{x}}_{t+\Delta t})  \\
    = &\frac{1}{(2\pi g^2(t) \Delta t)^{D/2}} \exp\left(-\frac{\|{\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t - {\mathbf{f}}({\mathbf{x}}_t, t) \Delta t\|_2^2}{2 g^2(t) \Delta t}\right)  \\
    &\qquad \cdot \exp\left(-\nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t) \cdot ({\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t) - \frac{\partial \log p_t({\mathbf{x}}_t)}{\partial t} \Delta t +\mathcal{O}(\Delta t)\right).
\end{aligned}
$$

**代数整理。** 关键步骤是对指数配方。我们有：

$$
\begin{aligned}
&-\frac{\|{\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t - {\mathbf{f}}({\mathbf{x}}_t, t) \Delta t\|_2^2}{2 g^2(t) \Delta t} - \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t) \cdot ({\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t)  \\
    = &-\frac{\left[\|{\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t - {\mathbf{f}}({\mathbf{x}}_t, t) \Delta t\|_2^2 + 2 g^2(t) \Delta t \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t) \cdot ({\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t)\right]}{2 g^2(t) \Delta t}
\end{aligned}
$$

令 $\boldsymbol{\delta} := {\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t$ 且 $\boldsymbol{\mu} := {\mathbf{f}}({\mathbf{x}}_t, t) \Delta t$。则：

$$
\begin{aligned}
&\|\boldsymbol{\delta} - \boldsymbol{\mu}\|_2^2 + 2 g^2(t) \Delta t \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t) \cdot \boldsymbol{\delta}  \\
    = &\|\boldsymbol{\delta}\|_2^2 - 2\boldsymbol{\delta} \cdot \boldsymbol{\mu} + \|\boldsymbol{\mu}\|_2^2 + 2 g^2(t) \Delta t \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t) \cdot \boldsymbol{\delta}  \\
    = &\|\boldsymbol{\delta}\|_2^2 - 2\boldsymbol{\delta} \cdot [\boldsymbol{\mu} - g^2(t) \Delta t \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t)] + \|\boldsymbol{\mu}\|_2^2  \\
    = &\|\boldsymbol{\delta} - [\boldsymbol{\mu} - g^2(t) \Delta t \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t)]\|_2^2 - \|g^2(t) \Delta t \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t)\|_2^2
\end{aligned}
$$

代回后得到：

$$
\begin{aligned}
&\|\boldsymbol{\delta} - [{\mathbf{f}}({\mathbf{x}}_t, t) \Delta t - g^2(t) \Delta t \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t)]\|_2^2  \\
    = &\|{\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t - [{\mathbf{f}}({\mathbf{x}}_t, t) - g^2(t) \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t)] \Delta t\|_2^2.
\end{aligned}
$$

因此，

$$
\begin{aligned}
p({\mathbf{x}}_t | {\mathbf{x}}_{t+\Delta t})
    =~&\frac{1}{(2\pi g^2(t) \Delta t)^{D/2}}  \\
    \quad &\cdot \exp\left(-\frac{\|{\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t - [{\mathbf{f}}({\mathbf{x}}_t, t) - g^2(t) \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t)] \Delta t\|_2^2}{2 g^2(t) \Delta t}\right)  \\
    \quad &\cdot \exp(\mathcal{O}(\Delta t))  \\
    =~&\mathcal{N}\left({\mathbf{x}}_t; {\mathbf{x}}_{t+\Delta t} - [{\mathbf{f}}({\mathbf{x}}_t, t) - g^2(t) \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t)] \Delta t, g^2(t) \Delta t \mathbf{I}\right)  \\
    \quad &\cdot (1 + \mathcal{O}(\Delta t)).
\end{aligned}
$$

配方产生的附加项 $\|g^2(t) \Delta t \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t)\|_2^2$ 为 $\mathcal{O}((\Delta t)^2)$，可吸收到误差项中。同样，时间导数项 $\frac{\partial \log p_t({\mathbf{x}}_t)}{\partial t} \Delta t$ 为 $\mathcal{O}(\Delta t)$，将在连续极限中消失。

**取 $\Delta t \rightarrow 0$ 的极限。** 当 $\Delta t \approx 0$ 时，在光滑性假设下，以下近似成立：

$$
\begin{aligned}
{\mathbf{f}}({\mathbf{x}}_t, t) &\approx {\mathbf{f}}({\mathbf{x}}_{t+\Delta t}, t+\Delta t), \\
    g(t) &\approx g(t+\Delta t), \\
    \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t) &\approx \nabla_{{\mathbf{x}}} \log p_{t+\Delta t}({\mathbf{x}}_{t+\Delta t})  = {\mathbf{s}}({\mathbf{x}}_{t+\Delta t}, t+\Delta t).
\end{aligned}
$$

使用这些近似并作适当整理，得到：

$$
\begin{aligned}
&p({\mathbf{x}}_t | {\mathbf{x}}_{t+\Delta t}) \\ \approx &\frac{1}{(2\pi g^2(t) \Delta t)^{D/2}}  \exp \Bigg(
            \\ &-\frac{\Big\|{\mathbf{x}}_{t} - \Big({\mathbf{x}}_{t+\Delta t} - \big[{\mathbf{f}}({\mathbf{x}}_{t+\Delta t}, t+\Delta t) - g^2({t+\Delta t}) {\mathbf{s}}({\mathbf{x}}_{t+\Delta t}, t+\Delta t)\big] \Delta t\Big) \Big\|_2^2 }{2 g^2(t+\Delta t) \Delta t} \\
            &\Bigg).
\end{aligned}
$$

这意味着 $p({\mathbf{x}}_t | {\mathbf{x}}_{t+\Delta t})$ 近似为正态分布，满足：

$$
\begin{aligned}
\text{\textbf{Mean:}} &  \quad{\mathbf{x}}_{t+\Delta t} - \big[{\mathbf{f}}({\mathbf{x}}_{t+\Delta t}, t+\Delta t) - g^2(t+\Delta t) {\mathbf{s}} ({\mathbf{x}}_{t+\Delta t}, t+\Delta t)\big] \Delta t, \\
    \text{\textbf{Covariance:}} & \quad g^2(t+\Delta t) \Delta t \mathbf{I}.
\end{aligned}
$$

取 $\Delta t \rightarrow 0$ 时的极限，就“推导”出[式 (4.1.6)](01-score-sde-principles.md#eq-sde-backward)中的连续逆时间 SDE。

[^1]: 这一推导受到[这篇文章](https://alexxthiery.github.io/notes/reverse_and_tweedie/reverse_and_tweedie.html)所用方法的启发。