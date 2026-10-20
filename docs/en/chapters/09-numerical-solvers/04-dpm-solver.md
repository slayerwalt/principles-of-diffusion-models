# 9.4 DPM-Solver {#sec-dpm}

DPM-Solver ([Lu et al., 2022](../../references.md#ref-lu2022dpm)) develops higher-order solvers for the empirical PF-ODE by exploiting the same diffusion-specific structure that appeared in our discussion of DDIM. The central goal is to obtain accurate samples using far fewer neural-network evaluations than a first-order sampler. The original DPM-Solver focuses on noise prediction and constructs first-, second-, and third-order single-step solvers; later extensions, including DPM-Solver`++` ([Lu et al., 2022](../../references.md#ref-lu2022dpm2)) and DPM-Solver-v3 ([Zheng et al., 2023](../../references.md#ref-zheng2023dpm)), further develop this family.

**High-Level Idea of DPM-Solver.** The DDIM discussion in [Section 9.2](02-ddim.md#sec-revisit-ddim) showed that, under ${\boldsymbol{\epsilon}}$-prediction, a first-order update can be obtained by retaining the known diffusion schedule analytically while holding the neural prediction fixed over each step,

$$
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)
\approx
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s).
$$

DPM-Solver ([Lu et al., 2022](../../references.md#ref-lu2022dpm)) builds naturally on this idea: rather than treating the neural prediction as constant, it seeks higher-order approximations of how the prediction varies along the trajectory. Like DEIS, it starts from the semilinear ${\boldsymbol{\epsilon}}$-prediction form of the PF-ODE,

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

and its variation-of-constants representation introduced in the Prologue. Its key additional ingredient is to reparameterize time by the *half-log signal-to-noise ratio*, following the log-SNR parameterization of VDM ([Kingma et al., 2021](../../references.md#ref-kingma2021variational)),

<span id="eq-lambda"></span>

$$
\begin{aligned}
\lambda_t
:=
\log\frac{\alpha_t}{\sigma_t}.
\end{aligned}
\tag{9.4.2}
$$

This coordinate is directly related to DDIM’s schedule-aligned clock $\rho_t=\sigma_t/\alpha_t$ (discussed around [Equation (9.2.4)](02-ddim.md#eq-ddim-coordinate-ode-t)) through $\rho_t=e^{-\lambda_t}$, and turns the remaining model-dependent term into a particularly convenient exponentially weighted integral. Local Taylor approximations of the neural prediction in $\lambda$ can then be integrated analytically, providing a systematic route from the first-order DDIM update to higher-order diffusion solvers.

## 9.4.1 DPM-Solver’s Insight: Time Reparameterization via Log-SNR {#subsec-dpm-method}

We now make the log-SNR reparameterization in [Equation (9.4.2)](04-dpm-solver.md#eq-lambda) precise and derive the corresponding PF-ODE representation.

**Change-of-Variable to Log-SNR in PF-ODE.** Suppose that $\alpha_t>0$, $\sigma_t>0$, and the log-SNR coordinate $\lambda_t$ in [Equation (9.4.2)](04-dpm-solver.md#eq-lambda) is strictly monotone on the interval of interest. It then admits an inverse function $t_\lambda(\cdot)$ satisfying

$$
t=t_\lambda(\lambda(t)).
$$

For schedules with $\sigma_0=0$, one has $\lambda_t\to+\infty$ as $t\to0$, so formulas involving the endpoint $t=0$ are understood through the corresponding limit or through a small positive terminal time in a numerical implementation.

We then express the state, schedule, and neural prediction in the $\lambda$-coordinate. A hat ($\hat{\cdot}$) together with $\lambda$ indicates that the corresponding quantity is viewed in this reparameterized time coordinate. More precisely, define

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

Hence, along the empirical PF-ODE trajectory,

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

Since

$$
\frac{\mathop{}\!\mathrm{d}\lambda_t}{\mathop{}\!\mathrm{d} t}
=
\frac{\alpha_t'}{\alpha_t}
-
\frac{\sigma_t'}{\sigma_t},
$$

the PF-ODE in [Equation (9.4.1)](04-dpm-solver.md#eq-noise-alpha-sigma) can be written as

$$
\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}_t}{\mathop{}\!\mathrm{d} t}
=
\frac{\mathop{}\!\mathrm{d}\log\alpha_t}{\mathop{}\!\mathrm{d} t}{\mathbf{x}}_t
-
\sigma_t
\frac{\mathop{}\!\mathrm{d}\lambda_t}{\mathop{}\!\mathrm{d} t}
\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t,t).
$$

Applying the chain rule,

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

gives the PF-ODE in log-SNR time:

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

Applying the same exponential-integrator (variation-of-constants) technique to [Equation (9.4.4)](04-dpm-solver.md#eq-diffusion-ode-lambda-eps) gives the following exact solution.

<span id="exp-wgt-exact-sol"></span>

**Proposition 9.4.1: Exponentially Weighted Exact Solution**

Given an initial value ${\mathbf{x}}_s$ at time $s>0$, the exact empirical PF-ODE solution $\widetilde{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)$ at time $t\in(0,s]$ can be re-expressed as

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

When the endpoint satisfies $\sigma_0=0$, the case $t=0$ is understood through the corresponding limit.

**Proof for Proposition.**

From [Equation (9.4.4)](04-dpm-solver.md#eq-diffusion-ode-lambda-eps), the integrating factor associated with the linear term satisfies

$$
\exp\left(
\int_{\lambda_s}^{\lambda_t}
\frac{\mathop{}\!\mathrm{d}\log\alpha_\lambda}{\mathop{}\!\mathrm{d}\lambda}
\mathop{}\!\mathrm{d}\lambda
\right)
=
\frac{\alpha_t}{\alpha_s}.
$$

Therefore, the variation-of-constants formula gives

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

where the last equality follows from $\sigma_\lambda/\alpha_\lambda=e^{-\lambda}$.

The key numerical consequence is now explicit. In $\lambda$-time, the only model-dependent quantity in the exact solution appears inside

$$
\int_{\lambda_s}^{\lambda_t}
e^{-\lambda}
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda)\mathop{}\!\mathrm{d}\lambda.
$$

When the model output is locally approximated by a Taylor polynomial in $\lambda$, the resulting exponentially weighted polynomial terms can be integrated in closed form, yielding explicit higher-order update coefficients. Thus, as in the factored viewpoint of the Prologue, the known schedule dependence is retained analytically while the approximation is concentrated on the neural prediction.

**Intuition of Why Reparameterize Time?** For strictly monotone $\lambda(t)$, a local first-order change of variables gives

$$
|\Delta t|
\approx
\frac{|\Delta\lambda|}{|\lambda'(t)|}.
$$

Thus, if the numerical grid uses a fixed step size $|\Delta\lambda|$, the induced step in $t$ is smaller where $|\lambda'(t)|$ is larger (i.e., where $\lambda$ changes rapidly with $t$), and larger where $|\lambda'(t)|$ is smaller.

The reparameterization itself does not alter the PF-ODE solution path; it only changes how the same path is parameterized:

$$
\frac{\mathop{}\!\mathrm{d} \hat{\mathbf{x}}_\lambda}{\mathop{}\!\mathrm{d} \lambda}
=
\frac{1}{\lambda'(t)}
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}_t}{\mathop{}\!\mathrm{d} t}.
$$

Hence, where $|\lambda'(t)|$ is large, the same change of the trajectory with respect to $t$ is represented over a larger range of $\lambda$.

More importantly, $\lambda(t)$ measures progress through the diffusion process directly in terms of the relative balance between signal and noise. A fixed $\Delta\lambda$ corresponds to a fixed multiplicative change in $\alpha_t/\sigma_t$, or equivalently to a fixed additive change in log-SNR, whereas the same $\Delta t$ may correspond to very different changes in corruption level at different times. Therefore, choosing numerical nodes uniformly in $\lambda$ provides a natural way to redistribute resolution along the diffusion trajectory. The precise induced spacing in the original time variable $t$ depends on the chosen noise schedule. Two simple examples illustrate this effect:

-   **$(\alpha_t,\sigma_t)=(1-t,t)$.** This corresponds to the linear interpolation commonly used in flow matching. Then

    $$
    \lambda(t)
    =
    \log\frac{1-t}{t},
    \qquad
    \lambda'(t)
    =
    -\frac{1}{t(1-t)},
    $$

    so locally

    $$
    |\Delta t|
    \approx
    |\Delta\lambda|\,t(1-t).
    $$

    Hence, a uniform grid in $\lambda$ induces very small steps in $t$ near both endpoints $t\to0$ and $t\to1$, and larger steps around the middle of the trajectory.

-   **$(\alpha_t,\sigma_t)=(1,t)$.** This corresponds to the EDM parameterization ([Karras et al., 2022](../../references.md#ref-karras2022elucidating)) when the independent variable $t$ is taken directly as the noise level $\sigma_t$. Then

    $$
    \lambda(t)
    =
    \log\frac{1}{t},
    \qquad
    \lambda'(t)
    =
    -\frac{1}{t},
    $$

    and therefore

    $$
    |\Delta t|
    \approx
    |\Delta\lambda|\,t.
    $$

    In fact, uniform spacing in $\lambda$ implies

    $$
    t=e^{-\lambda},
    $$

    so the corresponding noise levels are geometrically spaced. Thus, the grid contains finer absolute spacing at small noise levels (high SNR) and coarser absolute spacing at large noise levels (low SNR).

## 9.4.2 Estimating the Integral with Taylor Expansion {#subsec-dpm-method-2}

DEIS and DPM-Solver exploit the same factored principle but obtain higher-order information in different ways. DEIS reuses predictions from previous completed steps to extrapolate the neural prediction over the next interval. DPM-Solver instead works in the log-SNR coordinate $\lambda$ and uses a local Taylor expansion of the neural prediction, with the required higher-order information obtained from additional evaluations within the current step. We now derive this single-step construction.

From [Equation (9.4.5)](04-dpm-solver.md#eq-analytic-solution), the exact empirical PF-ODE flow initialized at the current numerical state $\tilde{\mathbf{x}}_s$ is

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

where $\hat{\mathbf{x}}_{\lambda_s}=\tilde{\mathbf{x}}_s$ and $\hat{\mathbf{x}}_\lambda$ denotes the corresponding exact empirical PF-ODE trajectory over the current step. Therefore, after the known schedule-dependent factors have been handled analytically, the remaining numerical problem is to approximate

$$
\lambda
\longmapsto
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda)
$$

inside the weighted integral.

For $n\geq1$, we expand this neural prediction about $\lambda_s$:

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

where

$$
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}^{(k)}
(\hat{\mathbf{x}}_\lambda,\lambda)
:=
\frac{\mathop{}\!\mathrm{d}^k}{\mathop{}\!\mathrm{d}\lambda^k}
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda)
$$

denotes the $k$-th *total derivative along the empirical PF-ODE trajectory*. In particular,

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

so the derivative includes both the explicit dependence on $\lambda$ and the dependence through the evolving state.

Substituting the Taylor expansion into [Equation (9.4.6)](04-dpm-solver.md#eqn-analytic-solution-each-step) gives

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

where

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

The coefficients ${\color{orange}C_k}$ depend only on the log-SNR interval and can be evaluated analytically.

If the required trajectory derivatives were available, dropping the $\mathcal O(h^{n+1})$ remainder would give the corresponding formal $n$-th order update. DPM-Solver avoids explicitly differentiating the neural network by approximating these derivatives from additional model evaluations, as described in the next subsection.

**Example: DPM-Solver-1**

For $n=1$, the exact empirical flow satisfies

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

Dropping the $\mathcal O(h^2)$ remainder defines DPM-Solver-1:

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

This finite-step update is exactly the deterministic DDIM update, as shown in [Section 9.4.4](04-dpm-solver.md#subsec-ddim-dpm-solver-1).

DPM-Solver-$n$ with $n \geq 2$ requires evaluating the $k^{\text{th}}$-derivative ${\color{cyan}\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}^{(k)}(\hat{\mathbf{x}}_\lambda, \lambda)}$ for $k \leq n-1$. However, directly computing higher-order derivatives is computationally expensive in practice. [Lu et al. (2022)](../../references.md#ref-lu2022dpm) also propose efficient approximation methods for these derivatives, which will be detailed in the next subsection.

## 9.4.3 Implementation of DPM-Solver-$n$ {#subsec-dpm-n-method}

**DPM-Solver-$n$ with $n \geq 2$.** In practice, implementing a higher-order DPM-Solver-$n$ entails the following:

-   Precomputing the coefficients ${\color{orange}C_k}$;

-   Approximating the $k^{\text{th}}$ derivative ${\color{cyan}\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}^{(k)}(\hat{\mathbf{x}}_\lambda,\lambda)}$ for $k \leq n-1$ to circumvent the costly computation of exact higher-order derivatives—a challenge well-studied in the ODE literature ([Hochbruck and Ostermann, 2005](../../references.md#ref-hochbruck2005explicit); [Luan, 2021](../../references.md#ref-luan2021efficient)). One common strategy is finite difference approximation.

We now elaborate on the first two points.

**Precomputing ${\color{orange}C_k}$.** Let $s$ and $t$ denote the start and end times, respectively, and define $h:=\lambda_t-\lambda_s$. The Taylor expansion of the exact empirical PF-ODE flow takes the form

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

where, for $m\geq1$,

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

For the first three cases,

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

**Example: Third-Order Expansion with Exact Derivatives**

For $n=3$, [Equation (9.4.9)](04-dpm-solver.md#eq-analytic-expansion) becomes

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

**Approximating ${\color{cyan}\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}^{(k)}(\hat{\mathbf{x}}_\lambda,\lambda)}$ for $k \leq n-1$.** For $n \geq 2$, following the standard approach of single-step ODE solvers ([Atkinson et al., 2009](../../references.md#ref-atkinson2009numerical)), [Lu et al. (2022)](../../references.md#ref-lu2022dpm) use additional model evaluations within the current step to approximate the higher-order derivatives. We illustrate this idea for $n=2$.

Let $\gamma\in(0,1]$ specify a stage point in the log-SNR interval $[\lambda_s,\lambda_t]$. Given an estimate $\tilde{\mathbf{x}}_s$ at $s$, define

$$
s^{\mathrm{mid}}
=
t_\lambda\left(\lambda_s+\gamma h\right),
\qquad
h:=\lambda_t-\lambda_s.
$$

For $\gamma\in(0,1)$ this is an interior point, while $\gamma=1$ corresponds to the endpoint $t$. We first use a DPM-Solver-1 step to predict the state at this stage:

$$
{\mathbf{x}}^{\mathrm{mid}}
=
\frac{\alpha_{s^{\mathrm{mid}}}}{\alpha_s}\tilde{\mathbf{x}}_s
-
\sigma_{s^{\mathrm{mid}}}
\left(e^{\gamma h}-1\right)
\boldsymbol{\epsilon}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s).
$$

The difference between the model predictions at the starting and stage points then provides a finite-difference approximation of the first total derivative along the trajectory:

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

Substituting this approximation into the second-order expansion gives

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

Thus, dropping the $\mathcal O(h^3)$ remainder yields a second-order single-step update.

For the midpoint choice $\gamma=\tfrac12$, DPM-Solver-2 admits the simpler two-stage form in [Algorithm 5](04-dpm-solver.md#alg-dpm-solver-2). This update differs from [Equation (9.4.11)](04-dpm-solver.md#eq-dpm-solver-2-app) only by $\mathcal O(h^3)$ and therefore has the same second-order accuracy.

<section class="book-algorithm" id="alg-dpm-solver-2" aria-labelledby="alg-dpm-solver-2-title" markdown="1">

<div class="book-algorithm__title" id="alg-dpm-solver-2-title" markdown="span">
Algorithm 5: DPM-Solver-2 (with $\gamma=\tfrac12$).
</div>

<div class="book-algorithm__meta" markdown="1">

**Input:** initial value ${\mathbf{x}}_T$, time steps $\{t_i\}_{i=0}^M$, model ${\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}$

</div>

<div class="book-algorithm__scroll" tabindex="0" role="region" aria-label="Algorithm 5 steps" markdown="1">
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

**Remark.**

The finite difference above approximates the *total* $\lambda$-derivative of the neural prediction along the empirical PF-ODE trajectory. Under standard smoothness assumptions,

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

Here, the intermediate state is itself obtained by a first-order DPM-Solver-1 step, whose stage error is $\mathcal O(h^2)$; under smoothness of the model, this does not change the $\mathcal O(h)$ accuracy of the difference quotient.

In [Equation (9.4.11)](04-dpm-solver.md#eq-dpm-solver-2-app), this derivative approximation is multiplied by

$$
e^h-h-1=\mathcal O(h^2),
$$

so its induced error is $\mathcal O(h^3)$. Hence, the resulting method is second-order accurate for any fixed $\gamma\in(0,1]$.

This single-step construction uses two model evaluations: one at $(\tilde{\mathbf{x}}_s,s)$ and one at the predicted stage $({\mathbf{x}}^{\mathrm{mid}},s^{\mathrm{mid}})$. The choice of $\gamma$ does not change the formal second-order accuracy, but it changes the stage location and the error constant. In particular, $\gamma=\tfrac12$ places the second evaluation at the midpoint of the log-SNR interval, giving the commonly used midpoint version of DPM-Solver-2.

**Implementation Detail: Selection of Sampling Timesteps.** Since DPM-Solver constructs its local high-order approximation in log-SNR time $\lambda$, a natural choice is to place the numerical nodes uniformly in $\lambda$. For a finite terminal time $t_M>0$, this gives

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

If $\sigma_0=0$, then $\lambda_0=+\infty$, so in practice one instead uses a finite terminal log-SNR corresponding to a small positive time, with the endpoint handled by the corresponding limiting or final denoising step.

Although the derivation is carried out in $\lambda$-space, the pre-trained model and schedule $(\alpha_t,\sigma_t)$ are usually parameterized by the original diffusion time $t$. During sampling, each selected log-SNR node is therefore mapped back through

$$
t_i=t_\lambda(\lambda_{t_i})
$$

whenever the model is evaluated or the schedule coefficients are required; see, for example, [Algorithm 5](04-dpm-solver.md#alg-dpm-solver-2). Thus, $\lambda$ serves as the numerical integration coordinate, while the sampler continues to evaluate the original pre-trained diffusion model at its corresponding time labels.

## 9.4.4 DDIM $=$ DPM-Solver-1 {#subsec-ddim-dpm-solver-1}

The first-order DPM-Solver recovers DDIM exactly. Both methods start from an exact integral representation of the same noise-prediction PF-ODE and make the same approximation over one step:

$$
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)
\approx
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s).
$$

The remaining schedule-dependent terms are then integrated analytically. DDIM applies this construction directly in the schedule-aligned form derived earlier, whereas DPM-Solver first rewrites the integral using the log-SNR coordinate

$$
\lambda_t
=
\log\frac{\alpha_t}{\sigma_t}.
$$

At first order, this change of numerical clock does not alter the constant approximation of the model prediction. Consequently, the two derivations produce exactly the same finite-step update.

<span id="ddim-dpm"></span>

**Proposition 9.4.2: DDIM is DPM-Solver-1**

For a fixed diffusion schedule $(\alpha_t,\sigma_t)$, the deterministic DDIM update in [Equation (9.2.1)](02-ddim.md#eq-ddim-euler) is identical to the DPM-Solver-1 update in [Equation (9.4.8)](04-dpm-solver.md#eq-dpm-1).

**Proof for Proposition.**

By the definition of $\lambda$,

$$
\frac{\sigma_s}{\alpha_s}
=
e^{-\lambda_s},
\qquad
\frac{\sigma_t}{\alpha_t}
=
e^{-\lambda_t}.
$$

Let

$$
h
=
\lambda_t-\lambda_s.
$$

Then

$$
e^h
=
\frac{\alpha_t/\sigma_t}{\alpha_s/\sigma_s}
=
\frac{\alpha_t\sigma_s}{\sigma_t\alpha_s},
$$

and therefore

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

Substituting this identity into the DPM-Solver-1 update gives

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

which is exactly the deterministic DDIM update in [Equation (9.2.1)](02-ddim.md#eq-ddim-euler).

Thus, DPM-Solver-1 and DDIM are two derivations of the same first-order finite-step update. The log-SNR coordinate becomes consequential beyond first order: higher-order DPM-Solvers use it to approximate how the noise prediction changes along the trajectory, rather than treating that prediction as constant over the whole step.

## 9.4.5 Relation of DPM-Solver to Runge-Kutta-Type Solvers {#subsec-dpm-rk}

DPM-Solver-1 is exactly the DDIM update ([Section 9.4.4](04-dpm-solver.md#subsec-ddim-dpm-solver-1)), so its first-order behavior is already understood. The higher-order question is how DPM-Solver improves this one-evaluation update.

The connection is to Runge-Kutta (RK) methods. Euler uses only the vector field at the beginning of a step, whereas RK methods obtain additional information through *stages*, i.e., evaluations at states and times generated within the current step; see [Section A.1.5](../../appendices/a-differential-equations/01-ordinary-differential-equations.md#app-de-solver-examples). Higher-order DPM-Solvers follow the same principle: instead of holding one neural prediction fixed over the whole step, they introduce additional within-step model evaluations to capture its variation. The diffusion-specific difference is that the known schedule-dependent factors are retained analytically, while the staged approximation is applied only to the learned prediction.

**DPM-Solver-2 versus Plain Midpoint RK2.** We first compare DPM-Solver-2 with classical midpoint RK2 applied directly to the complete noise-prediction PF-ODE in log-SNR time from [Equation (9.4.4)](04-dpm-solver.md#eq-diffusion-ode-lambda-eps). For brevity, denote its complete right-hand side by ${\mathbf{F}}(\hat{\mathbf{x}}_\lambda,\lambda)$.

For the midpoint choice $\gamma=\tfrac12$, let

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

and abbreviate

$$
\hat{\boldsymbol{\epsilon}}_s
:=
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,\lambda_s)
=
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_s,s).
$$

Classical midpoint RK2 gives

$$
{\mathbf{x}}_{\mathrm{mid}}^{\mathrm{RK}}
=
\tilde{\mathbf{x}}_s
+
\frac{h}{2}
{\mathbf{F}}(\tilde{\mathbf{x}}_s,\lambda_s),
$$

followed by

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

Thus, plain RK2 applies its two-stage construction to the complete PF-ODE velocity, including both the known schedule-dependent terms and the neural prediction.

By contrast, midpoint DPM-Solver-2 in [Algorithm 5](04-dpm-solver.md#alg-dpm-solver-2) first constructs the stage

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

Denoting the corresponding midpoint prediction by

$$
\hat{\boldsymbol{\epsilon}}_{\mathrm{mid}}^{\mathrm{DPM2}}
:=
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}
\left(
{\mathbf{x}}_{\mathrm{mid}}^{\mathrm{DPM2}},
\lambda_{\mathrm{mid}}
\right),
$$

the full update is

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

The structural difference is immediate. Plain midpoint RK2 discretizes the complete PF-ODE velocity, whereas DPM-Solver-2 retains the known diffusion-dependent evolution analytically and uses its two model evaluations only to approximate the variation of the noise prediction.

Under standard smoothness assumptions, both are second-order methods for the same empirical PF-ODE in $\lambda$-time. Their intermediate states generally differ by $\mathcal O(h^2)$, and their complete one-step updates therefore satisfy

$$
\tilde{\mathbf{x}}_t^{\mathrm{DPM2}}
-
\tilde{\mathbf{x}}_t^{\mathrm{RK2}}
=
\mathcal O(h^3).
$$

Thus, DPM-Solver-2 follows the same two-stage midpoint principle as classical RK2, while exploiting the known diffusion structure before applying the numerical approximation.

**DPM-Solver-2 versus Exponential Midpoint RK2.** A closer comparison keeps the semilinear decomposition in [Equation (9.4.4)](04-dpm-solver.md#eq-diffusion-ode-lambda-eps) explicit and treats its known linear part analytically. The corresponding *exponential midpoint RK2* construction first takes an exponential-Euler stage to the midpoint:

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

After evaluating

$$
\hat{\boldsymbol{\epsilon}}_{\mathrm{mid}}^{\mathrm{ExpRK}}
:=
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}
\left(
{\mathbf{x}}_{\mathrm{mid}}^{\mathrm{ExpRK}},
\lambda_{\mathrm{mid}}
\right),
$$

the full exponential-midpoint update is

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

When the linear coefficient is constant, these integral coefficients reduce to the usual $\varphi_1$ coefficients of explicit exponential midpoint RK2.

DPM-Solver-2 uses the same midpoint staging principle, but exploits the finer noise-prediction factorization

$$
-\sigma_\lambda
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}(\hat{\mathbf{x}}_\lambda,\lambda).
$$

Since $\sigma_\lambda$ is also known from the diffusion schedule, DPM-Solver-2 retains it inside the integral and stages only the neural prediction. Its midpoint and full-step coefficients are therefore

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

respectively, recovering the update in [Algorithm 5](04-dpm-solver.md#alg-dpm-solver-2).

Hence, the three constructions differ in what is approximated by the RK stages:

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

The exponential-RK and DPM midpoint stages differ by

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

For the complete step, a Taylor expansion about $\lambda_{\mathrm{mid}}$ gives

$$
\int_{\lambda_s}^{\lambda_t}
\frac{\sigma_\lambda-\sigma_{\mathrm{mid}}}{\alpha_\lambda}
\mathop{}\!\mathrm{d}\lambda
=
\mathcal O(h^3),
$$

while smoothness of the neural prediction gives

$$
\hat{\boldsymbol{\epsilon}}_{\mathrm{mid}}^{\mathrm{DPM2}}
-
\hat{\boldsymbol{\epsilon}}_{\mathrm{mid}}^{\mathrm{ExpRK}}
=
\mathcal O(h^2).
$$

Consequently,

$$
\tilde{\mathbf{x}}_t^{\mathrm{DPM2}}
-
\tilde{\mathbf{x}}_t^{\mathrm{ExpRK2}}
=
\mathcal O(h^3).
$$

Thus, DPM-Solver-2 and exponential midpoint RK2 agree through second order, but DPM-Solver-2 retains one additional analytically known quantity, $\sigma_\lambda$, inside the weighted integral.

**Remark.**

The discussion above concerns the noise-prediction formulation in which DPM-Solver is derived. If a pre-trained model uses another prediction parameterization, its output may be converted pointwise to noise prediction through [Equation (6.3.1)](../06-unified-perspective/03-equivalent-parameterizations.md#eq-predictions-equivalence) at each stage. Just like the DDIM-case in [Section 9.2.2](02-ddim.md#subsec-ddim-classical-euler), this should not be confused with directly applying a classical RK method to the raw prediction field: pointwise conversion and finite-step discretization are separate operations.

**DPM-Solver and Runge–Kutta Methods**

DPM-Solver combines the within-step staging principle of Runge–Kutta methods with the analytic treatment of known diffusion dynamics. DPM-Solver-1 is exactly DDIM, while higher-order DPM-Solvers use additional within-step evaluations to capture the variation of the neural prediction. The connection to classical RK methods is structural rather than an equality of finite-step formulas: DPM-Solver retains the known diffusion-dependent weights analytically before approximating the learned prediction.
