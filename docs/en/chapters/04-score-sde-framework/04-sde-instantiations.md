# 4.4 Instantiations of SDEs {#sec-instances}

[Song et al. (2020)](../../references.md#ref-song2020score) categorize the drift term ${\mathbf{f}}({\mathbf{x}}, t)$ and the diffusion term $g(t)$ in the forward SDE into three types based on the behavior of the variance during evolution. Here, we focus on two commonly used types: the *<u>V</u>ariance <u>E</u>xplosion* (VE) SDE and *<u>V</u>ariance <u>P</u>reserving* (VP) SDE. While it is possible to design custom noise schedulers, their design can substantially influence empirical performance. [Table 4.1](04-sde-instantiations.md#tb-sde-instances) summarizes these two SDE instantiations.

<span id="tb-sde-instances"></span>

**Table 4.1: Summary of the forward SDEs**

|  | **VE SDE** | **VP SDE** |
| --- | --- | --- |
| $\mathbf{f}({\mathbf{x}}, t)$ | $\mathbf{0}$ | $-\frac{1}{2}\beta(t){\mathbf{x}}$ |
| $g(t)$ | $\sqrt{\frac{\mathop{}\!\mathrm{d} \sigma^2(t)}{\mathop{}\!\mathrm{d} t}}$ | $\sqrt{\beta(t)}$ |
| SDE | $\mathop{}\!\mathrm{d}{\mathbf{x}}(t) = g(t) \mathop{}\!\mathrm{d}{\mathbf{w}}(t)$ | $\mathop{}\!\mathrm{d}{\mathbf{x}}(t) = -\frac{1}{2}\beta(t) {\mathbf{x}}(t) \mathop{}\!\mathrm{d} t + \sqrt{\beta(t)} \mathop{}\!\mathrm{d}{\mathbf{w}}(t)$ |
| $p_t({\mathbf{x}}_t\vert {\mathbf{x}}_0)$ | $\mathcal{N}\left({\mathbf{x}}_t ; {\mathbf{x}}_0, \left(\sigma^2(t) - \sigma^2(0)\right){\mathbf{I}}\right)$ | $\mathcal{N}\left({\mathbf{x}}_t ; {\mathbf{x}}_0 e^{-\frac{1}{2}\int_{0}^{t} \beta(\tau)\mathop{}\!\mathrm{d} \tau}, {\mathbf{I}} - {\mathbf{I}} e^{-\int_{0}^{t} \beta(\tau)\mathop{}\!\mathrm{d} \tau}\right)$ |
| $p_{\mathrm{prior}}$ | $\mathcal{N}(\boldsymbol{0}, \sigma^2(T){\mathbf{I}})$ | $\mathcal{N}(\boldsymbol{0}, {\mathbf{I}})$ |

## 4.4.1 VE SDE {#ve-sde}

VE SDE has the following components:

-   **Drift Term:** A zero drift term $\mathbf{f}=\boldsymbol{0}$.

-   **Diffusion Term:** $g(t)=\sqrt{\frac{\mathop{}\!\mathrm{d} \sigma^2(t)}{\mathop{}\!\mathrm{d} t}}$ for some function $\sigma(t)$.

The forward SDE then takes the following form:

<span id="eq-vesde"></span>

$$
\mathop{}\!\mathrm{d}{\mathbf{x}}(t) =  \sqrt{\frac{\mathop{}\!\mathrm{d} \sigma^2(t)}{\mathop{}\!\mathrm{d} t}} \mathop{}\!\mathrm{d}{\mathbf{w}}(t).
\tag{4.4.1}
$$

Similarly, the results from [Section 4.4.3](04-sde-instantiations.md#subsec-scoresde-perturbation-kernel) imply the perturbation kernel for the VE SDE and suggest selecting an appropriate prior distribution:

-   **Perturbation Kernel:**

    $$
    \begin{aligned}
    p_t({\mathbf{x}}_t | {\mathbf{x}}_0) &= \mathcal{N}\left({\mathbf{x}}_t ; {\mathbf{x}}_0, \left(\sigma^2(t) - \sigma^2(0)\right){\mathbf{I}}\right)
    \end{aligned}
    $$

-   **Prior Distribution:** Assume that $\sigma(t)$ is an increasing function for $t \in [0,T]$ and that $\sigma^2(T) \gg \sigma^2(0)$. The prior distribution is given by:

    $$
    p_{\mathrm{prior}} := \mathcal{N}(\boldsymbol{0}, \sigma^2(T){\mathbf{I}}).
    $$

A typical instance of a VE SDE is NCSN with the following design:

$$
\sigma(t):=\sigma_{\text{min}}\left(\frac{\sigma_{\text{max}}}{\sigma_{\text{min}}} \right)^t,\quad \text{for } t\in(0,1],
$$

where $\sigma_{\text{min}}$ and $\sigma_{\text{max}}$ are pre-specified constants. Namely, the sequence of variances is designed as a geometric sequence. With this, NCSN is viewed as a discretized version of VE SDE, as discussed in [Section 4.1.1](01-score-sde-principles.md#subsec-scoresde-motivation).

## 4.4.2 VP SDE {#vp-sde}

Let $\beta\colon [0,T] \rightarrow \mathbb{R}_{\geq 0 }$ be a non-negative function of $t$. A VP SDE is defined with the following components:

-   **Drift Term:** A linear drift given by $\mathbf{f}({\mathbf{x}}, t) = -\frac{1}{2}\beta(t){\mathbf{x}}$.

-   **Diffusion Term:** $g(t) = \sqrt{\beta(t)}$.

Thus, the forward SDE is expressed as:

<span id="eq-forward-linear-sde"></span>

$$
\begin{aligned}
\mathop{}\!\mathrm{d}{\mathbf{x}}(t) = -\frac{1}{2}\beta(t) {\mathbf{x}}(t) \mathop{}\!\mathrm{d} t + \sqrt{\beta(t)} \mathop{}\!\mathrm{d}{\mathbf{w}}(t).
\end{aligned}
\tag{4.4.2}
$$

Using the results from [Section 4.4.3](04-sde-instantiations.md#subsec-scoresde-perturbation-kernel), we can derive the perturbation kernel for the VP SDE and select an appropriate prior distribution:

-   **Perturbation Kernel:**

    $$
    p_t({\mathbf{x}}_t|{\mathbf{x}}_0) = \mathcal{N}\left({\mathbf{x}}_t ; {\mathbf{x}}_0 e^{-\frac{1}{2}\int_{0}^{t} \beta(\tau)\mathop{}\!\mathrm{d} \tau}, {\mathbf{I}} - {\mathbf{I}} e^{-\int_{0}^{t} \beta(\tau)\mathop{}\!\mathrm{d} \tau}\right).
    $$

-   **Prior Distribution:** $p_{\mathrm{prior}} := \mathcal{N}(\boldsymbol{0}, {\mathbf{I}})$.

We remark that since the perturbation kernel is a Gaussian with a known mean and covariance, we can apply [Equation (D.2.5)](../../appendices/d-supplementary-proofs/02-score-based-perspective.md#eq-gaussian-score) to compute its score function.

A classic example of a VP SDE is the DDPM, where the noise schedule $\beta(t)$ is defined as:

$$
\beta(t) := \beta_{\text{min}} + t(\beta_{\text{max}} - \beta_{\text{min}}), \quad \text{for all } t \in [0,1].
$$

Here, $\beta_{\text{min}}$ and $\beta_{\text{max}}$ are pre-defined constants. With this, DDPM can be interpreted as a discretization of the VP SDE, as discussed in [Section 4.1.1](01-score-sde-principles.md#subsec-scoresde-motivation).

## 4.4.3 (Optional) How Is the Perturbation Kernel $p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$ Derived? {#subsec-scoresde-perturbation-kernel}

If the drift term in the forward SDE [Equation (4.1.3)](01-score-sde-principles.md#eq-sde-forward) is linear in ${\mathbf{x}}$, taking the form

$$
\begin{aligned}
{\mathbf{f}}({\mathbf{x}}, t) = f(t) {\mathbf{x}},
\end{aligned}
$$

for some scalar-valued, time-dependent function $f(t) \in \mathbb{R}$, then [Equation (4.1.3)](01-score-sde-principles.md#eq-sde-forward) becomes a linear SDE:

$$
\mathop{}\!\mathrm{d}{\mathbf{x}}(t) = f(t){\mathbf{x}}(t) \mathop{}\!\mathrm{d} t + g(t) \mathop{}\!\mathrm{d}{\mathbf{w}}(t).
$$

Even if the initial distribution $p_{\mathrm{data}}$ is non-Gaussian, the linearity of the drift ensures that the conditional process remains Gaussian. In particular, for $t>0$, the transition kernel admits the form:

$$
p_t( {\mathbf{x}}_t | {\mathbf{x}}_0 ) = \mathcal{N} \left( {\mathbf{x}}_t;  {\mathbf{m}}(t),  P(t) \mathbf{I}_D \right),
$$

where ${\mathbf{x}}_0 \sim p_{\mathrm{data}}$, and ${\mathbf{m}}(t) \in \mathbb{R}^D$, $P(t)\in\mathbb{R}_{\ge 0}$ denote the conditional mean and (scalar) variance given ${\mathbf{x}}_0$, defined as:

$$
{\mathbf{m}}(t) = \mathbb{E} \left[{\mathbf{x}}_t | {\mathbf{x}}(0) = {\mathbf{x}}_0\right], \qquad
P(t) \mathbf{I}_D = \operatorname{Cov} \left[{\mathbf{x}}_t | {\mathbf{x}}(0) = {\mathbf{x}}_0\right].
$$

These first and second moments evolve according to the following ODEs ([Särkkä and Solin, 2019](../../references.md#ref-sarkka2019applied)):

<span id="eq-mean-var-evolution"></span>

$$
\begin{aligned}
    \frac{\mathop{}\!\mathrm{d} {\mathbf{m}}(t)}{\mathop{}\!\mathrm{d} t} &= f(t) {\mathbf{m}}(t), \\
    \frac{\mathop{}\!\mathrm{d} P(t)}{\mathop{}\!\mathrm{d} t} &= 2f(t) P(t) + g^2(t),
\end{aligned}
\tag{4.4.3}
$$

provided that the initial mean ${\mathbf{m}}(0)$ and variance $P(0)$ are finite.

Since both ODEs are linear, they admit closed-form solutions via the integrating factor method. Given the initial condition ${\mathbf{x}}_0$, the mean and variance evolve as

<span id="eq-mean-var-solution"></span>

$$
\begin{aligned}
{\mathbf{m}}(t) = \mathcal{E}(0  \to  t) {\mathbf{x}}_0,
\qquad
P(t) = \int_{0}^{t} \mathcal{E}^{2}(s  \to  t)  g(s)^{2} \mathop{}\!\mathrm{d} s,
\end{aligned}
\tag{4.4.4}
$$

with ${\mathbf{m}}(0)={\mathbf{x}}_0$ and $P(0)=0$. Here $\mathcal{E}(s \to t)$ denotes the exponential integrating factor

$$
\mathcal{E}(s  \to  t) := \exp \left(\int_{s}^{t} f(u) \mathop{}\!\mathrm{d} u\right),
$$

which captures the accumulated effect of the drift from time $s$ to $t$. Consequently, the transition kernel $p_t({\mathbf{x}}_t | {\mathbf{x}}_0)$ also admits a closed-form expression.

We defer the justification that the conditional covariance of $p_t({\mathbf{x}}_t |{\mathbf{x}}_0)$ is isotropic, that is $\operatorname{Cov}[{\mathbf{x}}_t | {\mathbf{x}}_0] = P(t) \mathbf I_D$ under a $D$-dimensional Wiener process with independent coordinates and diffusion $g(t)\mathbf I_D$, as well as the derivation of [Equation (4.4.3)](04-sde-instantiations.md#eq-mean-var-evolution), to [Section C.1.5](../../appendices/c-ito-calculus/01-ito-formula.md#subsec-rigorous-proof-linear-sde), which relies on Itô calculus.

**Example: VE SDE’s Transition Kernel**

In the special case of VE SDE: ${\mathbf{f}} \equiv 0$ and $g(t) = \sqrt{\frac{\mathop{}\!\mathrm{d} \sigma^2(t)}{\mathop{}\!\mathrm{d} t}}$, the mean and covariance of the solution to the SDE evolve as follows.

**Mean.**

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{m}}(t)}{\mathop{}\!\mathrm{d} t} = 0,
\quad \text{with} \quad {\mathbf{m}}(0) = {\mathbf{x}}_0
 \Longrightarrow
{\mathbf{m}}(t) = {\mathbf{x}}_0.
$$

**Variance.**

$$
\frac{\mathop{}\!\mathrm{d} P(t)}{\mathop{}\!\mathrm{d} t} = \frac{\mathop{}\!\mathrm{d} \sigma^2(t)}{\mathop{}\!\mathrm{d} t},
\quad \text{with} \quad P(0) = 0
 \Longrightarrow
P(t) = \sigma^2(t) - \sigma^2(0).
$$

Therefore

$$
p_t({\mathbf{x}}_t | {\mathbf{x}}_0)=\mathcal{N} \left({\mathbf{x}}_t;  {\mathbf{x}}_0,  \big(\sigma^2(t)-\sigma^2(0)\big)\mathbf{I}_D\right).
$$

**Example: VP SDE’s Transition Kernel**

In the VP SDE case with drift ${\mathbf{f}}({\mathbf{x}}, t) = -\tfrac{1}{2}\beta(t) {\mathbf{x}}$ and diffusion $g(t) = \sqrt{\beta(t)}$:

**Mean ${\mathbf{m}}(t)$.**

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{m}}}{\mathop{}\!\mathrm{d} t}
= -\frac{1}{2}\beta(t) {\mathbf{m}}(t),
\quad
B(t) \coloneqq \int_{0}^{t} \beta(s) \mathop{}\!\mathrm{d} s,
\quad
{\mathbf{m}}(t) = e^{-\frac{1}{2}B(t)} {\mathbf{x}}_0.
$$

**Variance $P(t)$.** The variance satisfies

$$
\frac{\mathop{}\!\mathrm{d} P}{\mathop{}\!\mathrm{d} t} = -\beta(t) P(t) + \beta(t).
$$

Applying the integrating factor $e^{B(t)}$ with $B(t) = \int_0^t \beta(s) \mathop{}\!\mathrm{d} s$, we obtain

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} \left[P(t)e^{B(t)}\right] = \beta(t) e^{B(t)}.
$$

Integrating both sides gives

$$
P(t) = 1 - e^{-B(t)}.
$$

Hence the covariance is isotropic with

$$
{\mathbf{P}}(t) = P(t) \mathbf I_D = \bigl(1 - e^{-B(t)}\bigr)\mathbf I_D.
$$

**Final Closed-Form Transition Kernel.**

$$
p_t({\mathbf{x}}_t|{\mathbf{x}}_0)
= \mathcal{N} \left(
   {\mathbf{x}}_t;
   \underbrace{e^{-\frac{1}{2} B(t)} {\mathbf{x}}_0}_{{\mathbf{m}}(t)},
   \underbrace{\big(1 - e^{-B(t)}\big)\mathbf{I}_D}_{P(t)\mathbf{I}_D}
\right),
\quad
B(t) = \int_0^t \beta(s) \mathop{}\!\mathrm{d} s.
$$
