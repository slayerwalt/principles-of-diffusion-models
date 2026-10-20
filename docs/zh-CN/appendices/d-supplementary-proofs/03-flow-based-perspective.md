# D.3 基于流的视角 {#app-sec-flow-proof}

## D.3.1 [引理 5.1.1](../../chapters/05-flow-based-perspective/01-normalizing-flows-and-neural-odes.md#instant-change-of-var)：瞬时变量替换 {#instantaneous-change-of-variables}

***证明。***

**方法 1：变量替换公式。** 将 $p({\mathbf{x}}(t), t)$ 记为 $p_t({\mathbf{x}}_t)$。从 ODE 的离散化出发，

$$
{\mathbf{z}}_{t+\Delta t} = {\mathbf{z}}_t + \Delta t  {\mathbf{F}}({\mathbf{z}}_t, t),
$$

归一化流的变量替换公式（[式（5.1.1）](../../chapters/05-flow-based-perspective/01-normalizing-flows-and-neural-odes.md#eq-nf-change-of-var-log)）给出

$$
\begin{aligned}
\log p_{t+\Delta t}({\mathbf{z}}_{t+\Delta t})
    &= \log p_t({\mathbf{z}}_t) - \log \Big| \det\big({\mathbf{I}} + \Delta t \nabla_{{\mathbf{z}}} {\mathbf{F}}({\mathbf{z}}_t, t)\big) \Big| \\
    &= \log p_t({\mathbf{z}}_t) - \operatorname{Tr}\Big(\log({\mathbf{I}} + \Delta t \nabla_{{\mathbf{z}}} {\mathbf{F}}({\mathbf{z}}_t, t)) \Big) \\
    &= \log p_t({\mathbf{z}}_t) - \Delta t  \operatorname{Tr}\big(\nabla_{{\mathbf{z}}} {\mathbf{F}}({\mathbf{z}}_t, t)\big) + \mathcal{O}(\Delta t^2),
\end{aligned}
$$

其中使用了 $\log \det = \operatorname{Tr} \log$，以及 $\Delta t$ 很小时的展开。取极限 $\Delta t \to 0$，便得到对数密度的连续时间微分方程。事实上，[式（5.1.6）](../../chapters/05-flow-based-perspective/01-normalizing-flows-and-neural-odes.md#eq-log-det-tr)也使用了相同技巧。

**方法 2：连续性方程。** 也可以利用连续性方程，它本质上起到了变量替换公式的作用：

$$
\partial_t p({\mathbf{z}},t) = - \nabla_{{\mathbf{z}}} \cdot \big({\mathbf{F}}({\mathbf{z}},t) p({\mathbf{z}},t)\big).
$$

展开散度，

$$
\partial_t p = - \big( (\nabla_{{\mathbf{z}}} \cdot {\mathbf{F}}) p + {\mathbf{F}} \cdot \nabla_{{\mathbf{z}}} p \big).
$$

沿轨迹 ${\mathbf{z}}(t)$，若其满足 $\frac{\mathop{}\!\mathrm{d} {\mathbf{z}}}{\mathop{}\!\mathrm{d} t} = {\mathbf{F}}({\mathbf{z}}(t), t)$，则关于时间的全导数为

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} p({\mathbf{z}}(t), t)
&= \nabla_{{\mathbf{z}}} p \cdot \frac{\mathop{}\!\mathrm{d} {\mathbf{z}}}{\mathop{}\!\mathrm{d} t} + \partial_t p \\
&= \nabla_{{\mathbf{z}}} p \cdot {\mathbf{F}} - \big( (\nabla_{{\mathbf{z}}} \cdot {\mathbf{F}}) p + {\mathbf{F}} \cdot \nabla_{{\mathbf{z}}} p \big) \\
&= - (\nabla_{{\mathbf{z}}} \cdot {\mathbf{F}}) p.
\end{aligned}
$$

除以 $p({\mathbf{z}}(t), t)$，得到

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} \log p({\mathbf{z}}(t), t) = - \nabla_{{\mathbf{z}}} \cdot {\mathbf{F}}({\mathbf{z}}(t), t).
$$

$\blacksquare$

## D.3.2 [定理 5.2.2](../../chapters/05-flow-based-perspective/02-flow-matching-framework.md#thm-continuity-mass)：质量守恒判据 {#app-sec-mass-conservation}

**预备知识：流映射与流诱导的密度。** 对任意初始位置 $\mathbf{x}_0\in\mathbb{R}^D$，流映射 $\boldsymbol{\Psi}_t:\mathbb{R}^D\to\mathbb{R}^D$ 是如下 ODE 的唯一解：

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} \boldsymbol{\Psi}_t(\mathbf{x}_0)
= \mathbf{v}_t\bigl(\boldsymbol{\Psi}_t(\mathbf{x}_0)\bigr),
\quad \boldsymbol{\Psi}_0(\mathbf{x}_0)=\mathbf{x}_0.
$$

在所作的正则性假设下，$\boldsymbol{\Psi}_t$ 关于 $t$ 和 $\mathbf{x}_0$ 都连续可微。

流诱导的密度 $p_t^{\mathrm{fwd}}$ 是初始密度 $p_0$ 经 $\boldsymbol{\Psi}_t$ 推前得到的：

$$
p_t^{\mathrm{fwd}}(\mathbf{x})
= p_0\bigl(\boldsymbol{\Psi}_t^{-1}(\mathbf{x})\bigr)
  \Bigl|\det \bigl(\nabla\boldsymbol{\Psi}_t^{-1}(\mathbf{x})\bigr)\Bigr|.
$$

它给出了位置 $\mathbf{x}$、时间 $t$ 处的粒子密度，这些粒子从 $p_0=p_{\mathrm{data}}$ 出发，并在速度场 $\mathbf{v}_t$ 下演化。

***非形式化证明：*** **充分条件：$p_t^{\mathrm{fwd}} = p_t$ $\Rightarrow$ 连续性方程。** 在[第 B.1.2 节](../b-density-evolution/01-change-of-variable.md#subsec-cov-continuity-eq)中，通过对离散变量替换公式取连续时间极限，我们得到了连续性方程的*强解*。这种方法假设密度 $p_t$ 足够光滑，使得所有导数都在经典意义下存在，且 PDE 逐点成立。这里给出一种互补的*弱意义*推导：仅在与任意光滑测试函数相乘并积分后要求连续性方程成立，从而放宽对 $p_t$ 和速度场 $\mathbf{v}_t$ 的正则性要求。这种弱形式能够容纳正则性更低的解，因此更加严谨，也是 PDE 理论和数值分析中的标准框架。

对任意具有紧支撑的光滑测试函数 $\varphi(\mathbf{x})$，推前性质给出：

$$
\begin{aligned}
\int p_t^{\mathrm{fwd}}(\mathbf{x}) \varphi(\mathbf{x})  \mathop{}\!\mathrm{d}\mathbf{x} &= \int p_0(\boldsymbol{\Psi}_t^{-1}(\mathbf{x})) \left| \det\left( \nabla \boldsymbol{\Psi}_t^{-1}(\mathbf{x}) \right) \right| \varphi(\mathbf{x}) \mathop{}\!\mathrm{d}\mathbf{x}\\ &=  \int p_0(\mathbf{y}) \varphi(\boldsymbol{\Psi}_t(\mathbf{y})) \mathop{}\!\mathrm{d}\mathbf{y},
\end{aligned}
$$

其中第二个等式由变量替换 $\mathbf{x} = \boldsymbol{\Psi}_t(\mathbf{y})$ 得到，且 $\mathop{}\!\mathrm{d}\mathbf{y} = \left| \det\left( \nabla \boldsymbol{\Psi}_t^{-1}(\mathbf{x}) \right) \right| \mathop{}\!\mathrm{d}\mathbf{x}$。

两边对 $t$ 求导：

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} \int p_t^{\mathrm{fwd}}(\mathbf{x}) \varphi(\mathbf{x}) \mathop{}\!\mathrm{d}\mathbf{x} = \frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} \int p_0(\mathbf{y}) \varphi(\boldsymbol{\Psi}_t(\mathbf{y})) \mathop{}\!\mathrm{d}\mathbf{y}.
$$

左侧为：

$$
\int \frac{\partial p_t^{\mathrm{fwd}}}{\partial t}(\mathbf{x}) \varphi(\mathbf{x}) \mathop{}\!\mathrm{d}\mathbf{x}.
$$

右侧为：

$$
\int p_0(\mathbf{y}) \nabla \varphi(\boldsymbol{\Psi}_t(\mathbf{y})) \cdot \mathbf{v}_t(\boldsymbol{\Psi}_t(\mathbf{y})) \mathop{}\!\mathrm{d}\mathbf{y},
$$

因为 $\frac{\partial \boldsymbol{\Psi}_t}{\partial t}(\mathbf{y}) = \mathbf{v}_t(\boldsymbol{\Psi}_t(\mathbf{y}))$。进行变量替换 $\mathbf{x} = \boldsymbol{\Psi}_t(\mathbf{y})$，则

$$
\mathop{}\!\mathrm{d}\mathbf{y} = \left| \det\left( \nabla \boldsymbol{\Psi}_t^{-1}(\mathbf{x}) \right) \right| \mathop{}\!\mathrm{d}\mathbf{x} ,
$$

并且

$$
p_0(\mathbf{y}) = p_t^{\mathrm{fwd}}(\mathbf{x}) \left| \det\left( \nabla \boldsymbol{\Psi}_t(\mathbf{y}) \right) \right| = \frac{p_t^{\mathrm{fwd}}(\mathbf{x})}{\left| \det\left( \nabla \boldsymbol{\Psi}_t^{-1}(\mathbf{x}) \right) \right|}.
$$

于是，右侧变为：

$$
\int p_t^{\mathrm{fwd}}(\mathbf{x}) \nabla \varphi(\mathbf{x}) \cdot \mathbf{v}_t(\mathbf{x}) \mathop{}\!\mathrm{d}\mathbf{x}.
$$

利用 $\varphi$ 的紧支撑性质进行分部积分：

$$
\int p_t^{\mathrm{fwd}}(\mathbf{x}) \nabla \varphi(\mathbf{x}) \cdot \mathbf{v}_t(\mathbf{x}) \mathop{}\!\mathrm{d}\mathbf{x} = -\int \varphi(\mathbf{x}) \nabla \cdot (p_t^{\mathrm{fwd}}(\mathbf{x}) \mathbf{v}_t(\mathbf{x})) \mathop{}\!\mathrm{d}\mathbf{x}.
$$

令两侧相等：

$$
\int \left[ \frac{\partial p_t^{\mathrm{fwd}}}{\partial t}(\mathbf{x}) + \nabla \cdot (p_t^{\mathrm{fwd}}(\mathbf{x}) \mathbf{v}_t(\mathbf{x})) \right] \varphi(\mathbf{x}) \mathop{}\!\mathrm{d}\mathbf{x} = 0.
$$

由于 $\varphi$ 是任意的，得到：

$$
\frac{\partial p_t^{\mathrm{fwd}}}{\partial t} + \nabla \cdot (p_t^{\mathrm{fwd}} \mathbf{v}_t) = 0.
$$

结合 $p_t^{\mathrm{fwd}} = p_t$，可得：

$$
\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t \mathbf{v}_t) = 0.
$$

**必要条件：连续性方程 $\Rightarrow p_t^{\mathrm{fwd}} = p_t$。** 假设 $p_t$ 满足连续性方程：

$$
\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t \mathbf{v}_t) = 0,
$$

并具有初始条件 $p_0(\mathbf{x}) = p_{\mathrm{data}}(\mathbf{x})$。由上文可知，$p_t^{\mathrm{fwd}}$ 满足相同的连续性方程，并且：

$$
p_0^{\mathrm{fwd}}(\mathbf{x}) = p_0(\boldsymbol{\Psi}_0^{-1}(\mathbf{x})) \left| \det\left( \nabla \boldsymbol{\Psi}_0^{-1}(\mathbf{x}) \right) \right| = p_0(\mathbf{x}),
$$

因为 $\boldsymbol{\Psi}_0(\mathbf{x}) = \mathbf{x}$。因此，两种密度具有相同的初始条件 $p_0 = p_{\mathrm{data}}$。

连续性方程可改写为：

$$
\frac{\partial p}{\partial t} + \mathbf{v}_t \cdot \nabla p + p \nabla \cdot \mathbf{v}_t = 0.
$$

这是一阶线性 PDE。假设 $\mathbf{v}_t$ 连续可微且全局 Lipschitz 连续，$p_t$ 足够光滑，则特征线法保证光滑函数空间中的解唯一。由于 $p_t$ 和 $p_t^{\mathrm{fwd}}$ 满足相同的 PDE 与初始条件，可得：

$$
p_t(\mathbf{x}) = p_t^{\mathrm{fwd}}(\mathbf{x})
$$

对所有 $t\in[0,1]$ 和 $\mathbf{x} \in \mathbb{R}^D$ 均成立。

由此完成等价性的证明。$\blacksquare$
