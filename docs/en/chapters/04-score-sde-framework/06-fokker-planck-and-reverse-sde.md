# 4.6 (Optional) Fokker–Planck Equation and Reverse-Time SDEs via Marginalization and Bayes’ Rule {#eq-heuristic-fpe-sde}

In this section, we offer a probabilistic perspective on the structure of the Fokker–Planck equation and the reverse-time SDE. By leveraging fundamental tools such as the marginalization trick and Bayes’ rule, we illuminate the connection between the statistical formulation of stochastic processes and their corresponding differential equations.

We emphasize that the “derivation” presented here is not mathematically rigorous proofs, but rather heuristic arguments intended to convey the underlying connections.

## 4.6.1 Fokker-Planck Equation from the Marginalization of Transition Kernels {#subsec-fpe-reason}

Given the forward transition probability as in [Equation (4.1.2)](01-score-sde-principles.md#eq-transition-forward-f-g)

$$
\begin{aligned}
p(\mathbf{x}_{t+\Delta t} | \mathbf{x}_t) = \mathcal{N}\left(\mathbf{x}_{t+\Delta t}; \mathbf{x}_t + \mathbf{f}(\mathbf{x}_t, t)\Delta t, g^2(t)\Delta t \mathbf{I}\right),
\end{aligned}
$$

and the marginal distributions

$$
\begin{aligned}
p_t(\mathbf{x}_t), \quad p_{t+\Delta t}(\mathbf{x}_{t+\Delta t}),
\end{aligned}
$$

we aim to derive the Fokker-Planck equation that governs the time evolution of the marginal distribution $p_t$.

**Change of Variables.** By the Markov property, the marginal distribution at time $t + \Delta t$ can be expressed as an integral over the previous state $\mathbf{x}_t$ (i.e., Chapman-Kolmogorov equation):

$$
\begin{aligned}
p_{t+\Delta t}({\mathbf{x}})
= \int \mathcal N \big({\mathbf{x}}; {\mathbf{y}}+{\mathbf{f}}({\mathbf{y}},t)\Delta t, g^2(t)\Delta t {\mathbf{I}}\big) p_t({\mathbf{y}}) \mathop{}\!\mathrm{d}{\mathbf{y}}.
\end{aligned}
$$

We introduce a new variable

$$
{\mathbf{u}}  :=  {\mathbf{y}}+{\mathbf{f}}({\mathbf{y}},t)\Delta t,
$$

so the Gaussian is centered at ${\mathbf{u}}$. For small $\Delta t$, this map is invertible with

$$
{\mathbf{y}}  =  {\mathbf{u}}-{\mathbf{f}}({\mathbf{u}},t)\Delta t+\mathcal O(\Delta t^2),
\quad
\Bigg|\det\frac{\partial {\mathbf{y}}}{\partial {\mathbf{u}}}\Bigg|
 = 1-(\nabla_{\mathbf{u}} \cdot{\mathbf{f}})({\mathbf{u}},t) \Delta t+\mathcal O(\Delta t^2).
$$

Hence, change-of-variable formula leads us to:

$$
\begin{aligned}
p_{t+\Delta t}({\mathbf{x}})
&= \int \mathcal N\left({\mathbf{x}};{\mathbf{u}},g^2(t)\Delta t{\mathbf{I}}\right) \cdot
\\\quad\Big[p_t({\mathbf{u}})&-\Delta t {\mathbf{f}}({\mathbf{u}},t) \cdot \nabla_{\mathbf{u}} p_t({\mathbf{u}})
-\Delta t (\nabla_{\mathbf{u}} \cdot{\mathbf{f}})({\mathbf{u}},t) p_t({\mathbf{u}})\Big] \mathop{}\!\mathrm{d}{\mathbf{u}}
+\mathcal O(\Delta t^2),
\end{aligned}
$$

**Taylor Expansion.** For any smooth function $\phi:{\mathbb{R}}^D\to{\mathbb{R}}$ and scale $\sigma>0$, if ${\mathbf{z}}\sim\mathcal N(\boldsymbol{0},{\mathbf{I}})$, the following approximation holds (known as the *Taylor–Gaussian smoothing formula*):

$$
\int \mathcal N({\mathbf{x}};{\mathbf{u}},\sigma^2{\mathbf{I}}) \phi({\mathbf{u}}) \mathop{}\!\mathrm{d}{\mathbf{u}}
={\mathbb{E}} \left[\phi({\mathbf{x}}+\sigma{\mathbf{z}})\right]
=\phi({\mathbf{x}})+\frac{\sigma^2}{2} \Delta_{\mathbf{x}}\phi({\mathbf{x}})+\mathcal O(\sigma^4).
$$

This is because Taylor expansion for:

$$
\phi({\mathbf{x}}+\sigma{\mathbf{z}})=\phi({\mathbf{x}})+\sigma\nabla_{\mathbf{x}}\phi({\mathbf{x}}) \cdot {\mathbf{z}}+\frac{\sigma^2}{2}{\mathbf{z}}^\top\nabla_{\mathbf{x}}^2\phi({\mathbf{x}}){\mathbf{z}}+\mathcal O(\sigma^3)
$$

and ${\mathbb{E}}[{\mathbf{z}}]=\boldsymbol{0}$, ${\mathbb{E}}[{\mathbf{z}}{\mathbf{z}}^\top]={\mathbf{I}}$.

Apply this with $\phi=p_t$, $\phi={\mathbf{f}} \cdot \nabla_{\mathbf{u}} p_t$, and $\phi=(\nabla_{\mathbf{u}} \cdot{\mathbf{f}}) p_t$, and use $\sigma^2=g^2(t)\Delta t$, we can obtain

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

Divide by $\Delta t$ and let $\Delta t\to0$ to obtain the Fokker–Planck equation.

In [Section C.1.4](../../appendices/c-ito-calculus/index.md#subsec-rigorous-proof-fpe), we present the Itô–based derivation to complement the discrete–time view above.

## 4.6.2 Why Does Reverse-Time SDE Take The Form? {#subsec-reverse-sde-reason}

The rigorous derivation of the reverse-time SDE is technical and requires delving into the properties of the Fokker-Planck equation. However, the form of the reverse-time SDE can be understood intuitively through Bayes’ theorem. Here, we present a heuristic derivation to provide insight into why [Equation (4.1.6)](01-score-sde-principles.md#eq-sde-backward) takes its form, with the appearance of score functions[^1].

**Using Bayes’ Rule for Inversion.** Our goal is to determine the reverse-time transition kernel by first considering the discrete-time case:

$$
p({\mathbf{x}}_t | {\mathbf{x}}_{t+\Delta t}),
$$

and then taking $\Delta t \rightarrow 0$ to obtain the continuous-time formulation. Using Bayes’ theorem, we express:

<span id="eq-reverse-sde-bayes-discrete"></span>

$$
\begin{aligned}
p({\mathbf{x}}_t | {\mathbf{x}}_{t+\Delta t})
    &= p({\mathbf{x}}_{t+\Delta t} | {\mathbf{x}}_t) \frac{p_t({\mathbf{x}}_t)}{p_{t+\Delta t}({\mathbf{x}}_{t+\Delta t})}  \\
    &= p({\mathbf{x}}_{t+\Delta t} | {\mathbf{x}}_t) \exp\left(\log p_t({\mathbf{x}}_t) - \log p_{t+\Delta t}({\mathbf{x}}_{t+\Delta t})\right).
\end{aligned}
\tag{4.6.1}
$$

The forward transition kernel is assumed to be as in [Equation (4.1.2)](01-score-sde-principles.md#eq-transition-forward-f-g):

$$
p({\mathbf{x}}_{t+\Delta t} | {\mathbf{x}}_t) = \mathcal{N}\left({\mathbf{x}}_{t+\Delta t}; {\mathbf{x}}_t + {\mathbf{f}}({\mathbf{x}}_t, t)\Delta t, g^2(t)\Delta t \mathbf{I}\right)
$$

**Taylor Expansion.** To handle the exponential term, we apply a first-order Taylor expansion. The key insight is to expand around the point $({\mathbf{x}}_t, t)$ in both space and time:

$$
\begin{aligned}
\log p_{t+\Delta t}({\mathbf{x}}_{t+\Delta t}) &= \log p_t({\mathbf{x}}_t) + \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t) \cdot ({\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t)  \\
    &\quad + \frac{\partial \log p_t({\mathbf{x}}_t)}{\partial t} \Delta t + \mathcal{O}(\|{\mathbf{h}}\|_2^2)
\end{aligned}
$$

where ${\mathbf{h}} := ({\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t, \Delta t)$. Therefore:

<span id="eq-log-difference"></span>

$$
\begin{aligned}
\log p_t({\mathbf{x}}_t) - \log p_{t+\Delta t}({\mathbf{x}}_{t+\Delta t}) &= -\nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t) \cdot ({\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t)  \\
    &\quad - \frac{\partial \log p_t({\mathbf{x}}_t)}{\partial t} \Delta t + \mathcal{O}(\|{\mathbf{h}}\|_2^2)
\end{aligned}
\tag{4.6.2}
$$

For a diffusion process, ${\mathbf{x}}_{t+\Delta t}-{\mathbf{x}}_t=\mathcal O_p(\sqrt{\Delta t})$, so the second-order spatial Taylor terms are generally of order $\mathcal O_p(\Delta t)$. Accordingly, the following calculation should be understood as a leading-order heuristic.

**Substituting into the Reverse Transition.** Substituting equations [Equation (4.1.2)](01-score-sde-principles.md#eq-transition-forward-f-g) and [Equation (4.6.2)](06-fokker-planck-and-reverse-sde.md#eq-log-difference) into [Equation (4.6.1)](06-fokker-planck-and-reverse-sde.md#eq-reverse-sde-bayes-discrete):

$$
\begin{aligned}
&p({\mathbf{x}}_t | {\mathbf{x}}_{t+\Delta t})  \\
    = &\frac{1}{(2\pi g^2(t) \Delta t)^{D/2}} \exp\left(-\frac{\|{\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t - {\mathbf{f}}({\mathbf{x}}_t, t) \Delta t\|_2^2}{2 g^2(t) \Delta t}\right)  \\
    &\qquad \cdot \exp\left(-\nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t) \cdot ({\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t) - \frac{\partial \log p_t({\mathbf{x}}_t)}{\partial t} \Delta t +\mathcal{O}(\Delta t)\right).
\end{aligned}
$$

**Algebraic Manipulation.** The key step is to complete the square in the exponent. We have:

$$
\begin{aligned}
&-\frac{\|{\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t - {\mathbf{f}}({\mathbf{x}}_t, t) \Delta t\|_2^2}{2 g^2(t) \Delta t} - \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t) \cdot ({\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t)  \\
    = &-\frac{\left[\|{\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t - {\mathbf{f}}({\mathbf{x}}_t, t) \Delta t\|_2^2 + 2 g^2(t) \Delta t \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t) \cdot ({\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t)\right]}{2 g^2(t) \Delta t}
\end{aligned}
$$

Let $\boldsymbol{\delta} := {\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t$ and $\boldsymbol{\mu} := {\mathbf{f}}({\mathbf{x}}_t, t) \Delta t$. Then:

$$
\begin{aligned}
&\|\boldsymbol{\delta} - \boldsymbol{\mu}\|_2^2 + 2 g^2(t) \Delta t \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t) \cdot \boldsymbol{\delta}  \\
    = &\|\boldsymbol{\delta}\|_2^2 - 2\boldsymbol{\delta} \cdot \boldsymbol{\mu} + \|\boldsymbol{\mu}\|_2^2 + 2 g^2(t) \Delta t \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t) \cdot \boldsymbol{\delta}  \\
    = &\|\boldsymbol{\delta}\|_2^2 - 2\boldsymbol{\delta} \cdot [\boldsymbol{\mu} - g^2(t) \Delta t \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t)] + \|\boldsymbol{\mu}\|_2^2  \\
    = &\|\boldsymbol{\delta} - [\boldsymbol{\mu} - g^2(t) \Delta t \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t)]\|_2^2 - \|g^2(t) \Delta t \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t)\|_2^2
\end{aligned}
$$

Substituting back:

$$
\begin{aligned}
&\|\boldsymbol{\delta} - [{\mathbf{f}}({\mathbf{x}}_t, t) \Delta t - g^2(t) \Delta t \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t)]\|_2^2  \\
    = &\|{\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t - [{\mathbf{f}}({\mathbf{x}}_t, t) - g^2(t) \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t)] \Delta t\|_2^2.
\end{aligned}
$$

Therefore,

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

The additional term $\|g^2(t) \Delta t \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t)\|_2^2$ from completing the square is $\mathcal{O}((\Delta t)^2)$ and can be absorbed into the error term. Similarly, the time derivative term $\frac{\partial \log p_t({\mathbf{x}}_t)}{\partial t} \Delta t$ is $\mathcal{O}(\Delta t)$ and will vanish in the continuous limit.

**Taking $\Delta t \rightarrow 0$ Limit.** As $\Delta t \approx 0$, under smoothness assumptions, the following approximations hold:

$$
\begin{aligned}
{\mathbf{f}}({\mathbf{x}}_t, t) &\approx {\mathbf{f}}({\mathbf{x}}_{t+\Delta t}, t+\Delta t), \\
    g(t) &\approx g(t+\Delta t), \\
    \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t) &\approx \nabla_{{\mathbf{x}}} \log p_{t+\Delta t}({\mathbf{x}}_{t+\Delta t})  = {\mathbf{s}}({\mathbf{x}}_{t+\Delta t}, t+\Delta t).
\end{aligned}
$$

Using these approximations and some rearrangements, we obtain:

$$
\begin{aligned}
&p({\mathbf{x}}_t | {\mathbf{x}}_{t+\Delta t}) \\ \approx &\frac{1}{(2\pi g^2(t) \Delta t)^{D/2}}  \exp \Bigg(
            \\ &-\frac{\Big\|{\mathbf{x}}_{t} - \Big({\mathbf{x}}_{t+\Delta t} - \big[{\mathbf{f}}({\mathbf{x}}_{t+\Delta t}, t+\Delta t) - g^2({t+\Delta t}) {\mathbf{s}}({\mathbf{x}}_{t+\Delta t}, t+\Delta t)\big] \Delta t\Big) \Big\|_2^2 }{2 g^2(t+\Delta t) \Delta t} \\
            &\Bigg).
\end{aligned}
$$

This implies that $p({\mathbf{x}}_t | {\mathbf{x}}_{t+\Delta t})$ is roughly a normal distribution with:

$$
\begin{aligned}
\text{\textbf{Mean:}} &  \quad{\mathbf{x}}_{t+\Delta t} - \big[{\mathbf{f}}({\mathbf{x}}_{t+\Delta t}, t+\Delta t) - g^2(t+\Delta t) {\mathbf{s}} ({\mathbf{x}}_{t+\Delta t}, t+\Delta t)\big] \Delta t, \\
    \text{\textbf{Covariance:}} & \quad g^2(t+\Delta t) \Delta t \mathbf{I}.
\end{aligned}
$$

Taking the limit as $\Delta t \rightarrow 0$, we “derive” the reverse-time continuous SDE given in [Equation (4.1.6)](01-score-sde-principles.md#eq-sde-backward).

[^1]: This derivation is inspired by the approach in [this post](https://alexxthiery.github.io/notes/reverse_and_tweedie/reverse_and_tweedie.html).
