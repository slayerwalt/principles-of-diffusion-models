# A.2 Foundation of Stochastic Differential Equations {#app-sde-intro}

Stochastic Differential Equations (SDEs) are an extension of ordinary differential equations (ODEs) that incorporate randomness, providing a mathematical framework for modeling systems affected by uncertainty. This chapter introduces SDEs, beginning with the discretization of ODEs, extending to the discretization of SDEs, and culminating in a discussion of general SDEs, including Ito’s calculus and Ito’s formula.

## A.2.1 From ODEs to SDEs: An Intuitive Introduction {#from-odes-to-sdes}

Let us begin with a ODE describing the deterministic evolution of a state variable ${\mathbf{x}}(t) \in \mathbb{R}^D$:

<span id="eq-ode"></span>

$$
\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t} = {\mathbf{f}}({\mathbf{x}}(t), t), \quad {\mathbf{x}}(0) = {\mathbf{x}}_0.
\tag{A.2.1}
$$

Here, ${\mathbf{f}}: \mathbb{R}^D \times [0,T] \rightarrow \mathbb{R}^D$ is a time-dependent velocity field that governs the dynamics of ${\mathbf{x}}(t)$. The solution to this ODE is a smooth trajectory $t \mapsto {\mathbf{x}}(t)$, fully determined by the initial condition ${\mathbf{x}}_0$.

**Discretization Perspective.** To build intuition, consider an Euler discretization of [Equation (A.2.1)](02-stochastic-differential-equations.md#eq-ode) over small time steps $\Delta t$:

$$
{\mathbf{x}}_{t+\Delta t} = {\mathbf{x}}_t + {\mathbf{f}}({\mathbf{x}}_t, t) \Delta t.
$$

This approximation becomes more accurate as $\Delta t \to 0$, converging (under standard regularity conditions on ${\mathbf{f}}$) to the exact solution of the ODE.

**Introducing Randomness: From ODE to SDE.** In many real-world systems, perfect knowledge of the dynamics is unrealistic. Noise, uncertainty, or unmodeled interactions may affect the evolution. To incorporate such randomness, we augment the ODE with a stochastic term:

<span id="eq-euler-sde"></span>

$$
{\mathbf{x}}_{t+\Delta t} = {\mathbf{x}}_t + {\mathbf{f}}({\mathbf{x}}_t, t)\Delta t + g(t)\sqrt{\Delta t} \cdot \boldsymbol{\epsilon}_t,
\tag{A.2.2}
$$

where

-   $g: [0,T] \to \mathbb{R}$ is a diffusion coefficient (possibly dependent on both state and time, though here assumed time-dependent only),

-   $\boldsymbol{\epsilon}_t \sim \mathcal{N}(\boldsymbol{0}, {\mathbf{I}}_D)$ are i.i.d. standard Gaussian vectors.

This modified update rule reflects not just deterministic drift, but also random perturbations scaled by $\sqrt{\Delta t}$. The scaling ensures that the stochastic perturbation remains finite in the limit $\Delta t \to 0$. Importantly, this formulation gives rise to a *continuous-time stochastic process* as $\Delta t \to 0$, which leads us to the framework of SDE.

**Stochastic Differential Equations.** Formally, the limit of the discrete update [Equation (A.2.2)](02-stochastic-differential-equations.md#eq-euler-sde) as $\Delta t \to 0$ defines the SDE:

<span id="eq-sde"></span>

$$
\mathop{}\!\mathrm{d} {\mathbf{x}}(t) = {\mathbf{f}}({\mathbf{x}}(t), t)\mathop{}\!\mathrm{d} t + g(t) \mathop{}\!\mathrm{d} {\mathbf{w}}(t).
\tag{A.2.3}
$$

Here, ${\mathbf{w}}(t) \in \mathbb{R}^D$ is a *Wiener process* (standard Brownian motion), a continuous-time stochastic process characterized by:

-   **Initial State:** ${\mathbf{w}}(0) = 0$ almost surely;

-   **Independent Increments:** for $0 \leq s < t$, the increment ${\mathbf{w}}(t) - {\mathbf{w}}(s)$ is independent of the past;

-   **Gaussian Increments:**

    <span id="eq-wiener-gaussian"></span>

    $$
    \begin{aligned}
    {\mathbf{w}}(t) - {\mathbf{w}}(s) \sim \mathcal{N}\bigl(\boldsymbol{0}, (t - s){\mathbf{I}}_D \bigr)
    \end{aligned}
    \tag{A.2.4}
    $$

-   **Continuity:** Sample paths $t \mapsto {\mathbf{w}}(t)$ are almost surely continuous but nowhere differentiable.

In addition, the notation

$$
\mathop{}\!\mathrm{d} {\mathbf{w}}(t) := {\mathbf{w}}(t + \mathop{}\!\mathrm{d} t) - {\mathbf{w}}(t)
$$

is often used to denote the infinitesimal increment of the Wiener process.

While suggestive, this notation is heuristic and should not be interpreted as a classical differential (e.g., in the Riemann or Lebesgue sense), since Brownian paths are almost surely nowhere differentiable. Instead, it serves as a formal shorthand to express the Gaussian increments property:

$$
\mathop{}\!\mathrm{d} {\mathbf{w}}(t) \sim \mathcal{N}(0, \mathop{}\!\mathrm{d} t\, \mathrm{I}_D),
$$

meaning that over an infinitesimal time interval of length $\mathop{}\!\mathrm{d} t$, the increment of the Wiener process behaves like a Gaussian random variable with zero mean and covariance $\mathop{}\!\mathrm{d} t\, \mathrm{I}_D$.

## A.2.2 Further Explanation of [Equation (A.2.3)](02-stochastic-differential-equations.md#eq-sde) {#further-explanation-of-sde}

The SDE in [Equation (A.2.3)](02-stochastic-differential-equations.md#eq-sde) should be understood in its *integral form*:

<span id="eq-sde-integral-form"></span>

$$
\begin{aligned}
{\mathbf{x}}(t) = {\mathbf{x}}(0) + \int_0^t {\mathbf{f}}({\mathbf{x}}(s), s) \mathop{}\!\mathrm{d} s + \int_0^t g(s) \mathop{}\!\mathrm{d} {\mathbf{w}}(s),
\end{aligned}
\tag{A.2.5}
$$

interpreted in the *Itô sense*. Here, the first term is a classical (Riemann or Lebesgue) integral representing the accumulated deterministic drift, while the second term is an *Itô stochastic integral*, which integrates with respect to the Wiener process ${\mathbf{w}}(t)$. We do not provide a full rigorous construction of the Itô integral, but offer the following intuition.

**Intuition for Itô Integration.** The Itô integral can be understood as the limit of discrete sums of the form[^1]

$$
\sum_i g(t_i)\bigl(w(t_{i+1})-w(t_i)\bigr),
$$

where the integrand $g(t)$ is evaluated at the *left* endpoint $t_i$ of each subinterval. As the partition becomes finer, these sums converge to the Itô integral. The left-point evaluation rule is what distinguishes Itô integration from classical integrals, which may use midpoints or other evaluation rules. Because Brownian paths are continuous yet almost surely nowhere differentiable, classical integration does not directly apply. The Itô integral handles this irregularity and captures the cumulative effect of stochastic fluctuations over time.

**Use of Differential Notation.** Expressions such as $\mathop{}\!\mathrm{d} {\mathbf{x}}(t)$, $\mathop{}\!\mathrm{d} t$, and $\mathop{}\!\mathrm{d} {\mathbf{w}}(t)$ are not classical differentials. Instead, they are formal notations representing infinitesimal increments of the respective processes. While heuristic, they are widely used for their convenience in expressing SDEs analogously to ODEs and facilitate formal manipulations within Itô calculus.

How Itô calculus is applied in diffusion models will be explained in [Appendix C](../c-ito-calculus/index.md#app-ito).

**Comparison with ODEs.** In ODEs, e.g.,

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t} = {\mathbf{f}}({\mathbf{x}}(t), t),
$$

the integral form

$$
{\mathbf{x}}(t) = {\mathbf{x}}(0) + \int_0^t {\mathbf{f}}({\mathbf{x}}(\tau), \tau) \mathop{}\!\mathrm{d} \tau
$$

is justified by the *Fundamental Theorem of Calculus*, which ensures that differentiable functions can be recovered from their derivatives.

By contrast, in SDEs such as [Equation (A.2.3)](02-stochastic-differential-equations.md#eq-sde), there is no direct analog of this theorem because Brownian motion lacks differentiability, and stochastic integrals do not follow the classical chain rule. Instead, Itô calculus introduces alternative tools (e.g., Itô’s lemma) to analyze and manipulate stochastic dynamics.

Thus, while the differential notation for SDEs is compact and intuitive, a rigorous understanding depends on interpreting them via their integral formulation using Itô integrals.

## A.2.3 A Numerical Solver for SDE. {#a-numerical-solver-for-sde}

Like ODEs, the SDE in [Equation (A.2.3)](02-stochastic-differential-equations.md#eq-sde) admits a unique solution[^2] if $\mathbf{f}(\cdot, t)$ and $g(\cdot)$ satisfy some smoothness conditions: $\mathbf{f}(\cdot, t)$ is Lipschitz and of linear growth in $\mathbf{x}$, and $g(\cdot)$ is square integrable.

For general SDEs as in [Equation (A.2.3)](02-stochastic-differential-equations.md#eq-sde), closed-form solutions are generally unavailable, so numerical methods are necessary. A common approach is the *Euler–Maruyama method*, which generalizes Euler’s method for ODEs and, indeed, we have already seen it in [Equation (A.2.2)](02-stochastic-differential-equations.md#eq-euler-sde). It approximates the drift term ${\mathbf{f}}({\mathbf{x}}(t), t)$ over a time step $\Delta t$ and simulates the stochastic noise $g(t) \mathop{}\!\mathrm{d} {\mathbf{w}}(t)$ using Gaussian increments $\sqrt{\Delta t} \, \boldsymbol{\epsilon}_t$ with $\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

Later, in [Section C.1.5](../c-ito-calculus/01-ito-formula.md#subsec-rigorous-proof-linear-sde), we will see that a linear SDE admits a closed-form solution.

[^1]: For square-integrable integrands, this convergence holds in the $L^2$ sense (mean-square), which in turn implies convergence in probability. The $L^2$ viewpoint is natural because it connects directly to Itô’s isometry, the key identity underlying the standard and rigorous construction of the Itô integral. This isometry links stochastic integrals to standard integrals in expectation (${\mathbb{E}}\bigl[\bigl\|\int_0^T \boldsymbol{\psi}(t)\,\mathrm{d}{\mathbf{w}}_t \bigr\|^2\bigr] = {\mathbb{E}}\bigl[\int_0^T \|\boldsymbol{\psi}(t)\|_F^2\,\mathrm{d}t\bigr]$ with Frobenius norm $\|\boldsymbol{\psi}(t)\|_F^2$); we state and use it in [Section C.1.5](../c-ito-calculus/01-ito-formula.md#subsec-rigorous-proof-linear-sde).

[^2]: The solution is in the *strong* sense, meaning that $\mathbf{x}(t)$ satisfies the SDE in its integral form (see [Equation (A.2.5)](02-stochastic-differential-equations.md#eq-sde-integral-form)) with respect to the given Brownian motion $\mathbf{w}(t)$ on a fixed probability space. We omit the detailed technical definitions here.
