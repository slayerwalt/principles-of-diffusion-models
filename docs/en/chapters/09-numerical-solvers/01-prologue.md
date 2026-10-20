# 9.1 Prologue {#sec-prologue-solvers}

## 9.1.1 Fast Sampling as Numerical Integration {#fast-sampling-as-numerical-integration}

The Score SDE framework ([Song et al., 2020](../../references.md#ref-song2020score)) established a key foundation by linking discrete-time diffusion models and ELBO-based formulations ([Sohl-Dickstein et al., 2015](../../references.md#ref-sohl2015deep); [Ho et al., 2020](../../references.md#ref-ho2020denoising)) with continuous-time SDE/ODE generative modeling. This connection gives a clean way to think about sampling: after a diffusion model has been trained, generation can be viewed as solving a learned differential equation from noise back to data.

The practical difficulty is speed. A naive sampler follows this reverse-time trajectory with many small numerical steps, often requiring hundreds or thousands of neural network evaluations. Fast sampling asks whether we can approximate the same generation process with far fewer model calls.

This chapter focuses on *training-free* acceleration: the pre-trained diffusion model is kept fixed, and we design better numerical solvers for sampling. Training-based acceleration methods, such as distillation or direct few-step training, are discussed later in [Chapter 10](../10-distillation/index.md#ch-distillation) and [Chapter 11](../11-fast-generators/index.md#ch-fast-scratch).

The main message of this chapter is therefore simple:

**Message**

Once a diffusion model has been trained, training-free fast sampling is largely a numerical integration problem.

The literature contains many named diffusion samplers, which can make the subject appear more fragmented than it really is. In this chapter, we first introduce the small number of numerical ideas underlying these methods. Only afterward do we attach the familiar method names to them. This organization makes clear that the samplers are not unrelated tricks: they are different ways of approximating the same PF-ODE trajectory.

**Sampling as Solving the Empirical PF-ODE.** Suppose we have a pre-trained diffusion model

$$
{\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}},t)
    \approx
    \nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}}),
$$

or equivalently a model written in one of the other standard parameterizations from [Section 6.3](../06-unified-perspective/03-equivalent-parameterizations.md#sec-equivalent-parametrizations). The empirical PF-ODE uses this learned model in place of the true score:

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t}
    =
    {\mathbf{f}}({\mathbf{x}}(t),t)
    -
    \frac{1}{2}g^2(t)\,
    {\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}(t),t).
$$

Starting from ${\mathbf{x}}(T)\sim p_{\mathrm{prior}}$, generation integrates this ODE backward from $t=T$ to $t=0$.

For two times $s>t$, the exact empirical update can be written in integral form as

<span id="eq-solution-int-score"></span>

$$
\begin{aligned}
    \widetilde{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
    =
    {\mathbf{x}}_s
    +
    \int_s^t
    \left[
        {\mathbf{f}}({\mathbf{x}}(\tau),\tau)
        -
        \frac{1}{2}g^2(\tau)
        {\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}(\tau),\tau)
    \right]\mathop{}\!\mathrm{d} \tau .
\end{aligned}
\tag{9.1.1}
$$

Here, $\widetilde{{\boldsymbol{\Psi}}}_{s\to t}$ denotes the flow map of the empirical PF-ODE, obtained by replacing the oracle score with the learned model. As throughout this book, a *flow map* refers to the classical ODE solution map introduced in [Equation (4.2.2)](../04-score-sde-framework/02-matching-marginals.md#eq-flow-map-def). We denote the corresponding oracle flow map by ${\boldsymbol{\Psi}}_{s\to t}$. The empirical flow map is intended to approximate the oracle map:

$$
\widetilde{{\boldsymbol{\Psi}}}_{s\to t}({\mathbf{x}}_s)
    \approx
    {\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s).
$$

Because the integral in [Equation (9.1.1)](01-prologue.md#eq-solution-int-score) cannot generally be evaluated in closed form, a numerical sampler discretizes the reverse-time interval using a decreasing grid

<span id="eq-time-step-def"></span>

$$
\begin{aligned}
T=t_0>t_1>\cdots>t_M=0,
\end{aligned}
\tag{9.1.2}
$$

starts from

$$
\tilde{{\mathbf{x}}}_{t_0}={\mathbf{x}}_T\sim p_{\mathrm{prior}},
$$

and produces a sequence

$$
\tilde{{\mathbf{x}}}_{t_0},\tilde{{\mathbf{x}}}_{t_1},\ldots,\tilde{{\mathbf{x}}}_{t_M},
$$

where $\tilde{{\mathbf{x}}}_{t_i}$ approximates $\widetilde{\boldsymbol{\Psi}}_{T\to t_i}({\mathbf{x}}_T)$. The final iterate $\tilde{{\mathbf{x}}}_{t_M}$ is used as the generated sample.

Thus, every training-free solver in this chapter answers the same question:

<span id="question-9-1-1"></span>

**Question 9.1.1**

How should we approximate the PF-ODE integral accurately using as few model evaluations as possible?

**Three Independent Choices in Solver Design.** Once the decreasing time grid in [Equation (9.1.2)](01-prologue.md#eq-time-step-def) has been introduced, it is useful to distinguish three choices that are often intertwined in diffusion sampling.

1.  **Prediction Parameterization.** Which quantity is provided by the pre-trained diffusion model: score ${\mathbf{s}}_{\boldsymbol{\phi}^\times}$, noise ${\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}$, clean data ${\mathbf{x}}_{\boldsymbol{\phi}^\times}$, or velocity ${\mathbf{v}}_{\boldsymbol{\phi}^\times}$?

2.  **Time Parameterization.** Which variable is used to parameterize progress along the same continuous trajectory? The original diffusion time $t$ is one choice; other reparameterizations will be introduced later when they become useful.

3.  **Numerical Approximation.** How does the solver approximate the unknown model-dependent quantity over a finite interval? It may use only the current evaluation, obtain additional evaluations within the current step, or reuse evaluations from previous steps.

This separation will be important throughout the chapter. The first two choices determine the numerical form presented to the solver, while the third determines how the solver uses the available model evaluations.

**How Can a Solver Use Model Evaluations?** A useful way to organize numerical solvers is by asking where the information for the next update comes from. Starting from the current state, a solver may either obtain additional information *within the current step* or reuse information from *previous steps*.

A *single-step method* constructs the next state using only the current state and evaluations generated within the same time interval. Euler’s method is the simplest example, using only the current velocity. Higher-order Runge-Kutta (RK) methods may instead evaluate the velocity at one or more intermediate states before completing the step. Thus, “single-step” does not mean “one function evaluation”; it means that no information from earlier completed steps is required.

A *multistep method* reuses information from previous completed steps. Stored model evaluations provide information about how the dynamics have recently changed, allowing higher-order updates without several new evaluations within the current step. The Adams–Bashforth (AB) family is a representative classical example.

A separate idea is to change how updates across time are scheduled. Rather than computing all time points strictly sequentially, one may initialize states at several times and repeatedly correct them until they become consistent with the ODE. Such fixed-point iterations can allow model evaluations at different time points to be carried out in parallel.

Representative classical examples, including Euler, RK, AB, and Picard iteration, are reviewed in [Section A.1.5](../../appendices/a-differential-equations/01-ordinary-differential-equations.md#app-de-solver-examples); their structural organization is summarized in [Equation (A.1.6)](../../appendices/a-differential-equations/01-ordinary-differential-equations.md#eq-classic-solvers).

**Computational Cost and Practical Performance.** A numerical time step need not correspond to a single neural-network evaluation. Euler uses one evaluation per step, whereas higher-order RK methods may require several. Multistep methods instead reuse stored evaluations and, after initialization, can often advance using only one new evaluation per step.

We therefore measure sampling cost by the *number of function evaluations* (NFE). If a method uses approximately $m$ model evaluations per step over $M$ steps, then

$$
\mathrm{NFE}\approx mM.
$$

For instance, classifier-free guidance (see [Section 8.3](../08-guidance/03-classifier-free-guidance.md#sec-cfg)) requires both conditional and unconditional model predictions and therefore roughly doubles the neural-network evaluation cost per sampling step.

This classical taxonomy tells us *where numerical information comes from* and how many model evaluations it costs. Diffusion PF-ODEs offer an additional opportunity: not every part of the vector field is equally unknown. Depending on the prediction parameterization, part of the dynamics is determined analytically by the noise schedule, while only the remaining quantity requires a neural-network evaluation. We next examine how exposing this separation changes the numerical form of the PF-ODE and motivates diffusion-specific integrators.

## 9.1.2 Design of Diffusion-Specific Solvers {#subsec-common-design-solvers}

The classical numerical methods reviewed in [Section A.1.4](../../appendices/a-differential-equations/01-ordinary-differential-equations.md#app-de-solver) treat a general ODE through evaluations of its vector field. Diffusion PF-ODEs offer additional structure: the noise schedule is known analytically, whereas the diffusion-model prediction is supplied by a neural network.

This creates a useful opportunity. Rather than numerically approximating every part of the dynamics in the same way, a solver may retain the known schedule-dependent terms analytically and approximate only the model-dependent prediction. How clearly this structure is exposed depends on the prediction parameterization.

**I. Parameterizations Beyond the Score.** Although the PF-ODE can be written in score form, modern implementations often use noise prediction, clean-data prediction, or velocity prediction. These parameterizations are algebraically related, but they expose different numerical structures.

There is also an important stability reason to avoid direct score prediction in many solvers. As $t\to 0$, the true score $\nabla_{{\mathbf{x}}}\log p_t(\cdot)$ can change very rapidly, especially when the data distribution is concentrated near a low-dimensional manifold ([Kim et al., 2022](../../references.md#ref-kim2022soft)). This makes it difficult for a neural network trained to approximate the score directly to remain accurate near small noise levels [^1].

For this reason, a widely used alternative is to predict the noise ${\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}$, or related targets such as clean-data prediction or velocity prediction. In the noise-prediction form, the score approximation is

$$
{\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}},t)
    =
    -\frac{1}{\sigma_t}\,
    {\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}},t).
$$

These prediction parameterizations are algebraically related, but the quantity supplied by the neural network is only one component of the solver design. We next examine how the same PF-ODE appears numerically under these different parameterizations.

**II. One PF-ODE, Two Numerical Forms.** Once the prediction parameterization has been chosen, the same PF-ODE can be presented to a numerical solver in two particularly useful forms.

**Direct Velocity Form.** The most direct representation is

<span id="eq-direct-pfode-form"></span>

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t}
=
{\mathbf{v}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}(t),t).
\tag{9.1.3}
$$

Here, under ${\mathbf{v}}$-prediction, the model directly supplies the complete empirical PF-ODE velocity ${\mathbf{v}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}(t),t)$. Numerical integration methods therefore discretize the full right-hand side as a whole.

**Factored Semilinear Form.** Score, noise, and clean-data prediction naturally expose additional schedule-dependent structure. In each case, the PF-ODE can be written in the generic *semilinear* form

<span id="eq-general-semilinear-pfode"></span>

$$
\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t}
=
\underbrace{
a(t){\mathbf{x}}(t)
}_{\substack{\text{known}\\\text{linear part}}}
+
\underbrace{
b(t)
}_{\substack{\text{known}\\\text{coefficient}}}
\underbrace{
{\mathbf{N}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}(t),t)
}_{\substack{\text{neural}\\\text{evaluation}}},
\quad
{\mathbf{N}}_{\boldsymbol{\phi}^\times}
\in
\left\{
{\mathbf{s}}_{\boldsymbol{\phi}^\times},
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times},
{\mathbf{x}}_{\boldsymbol{\phi}^\times}
\right\}.
\tag{9.1.4}
$$

Here, $a(t)$ and $b(t)$ are known functions determined by the noise schedule, whereas ${\mathbf{N}}_{\boldsymbol{\phi}^\times}$ denotes the neural prediction[^2]. The term *semilinear* refers to this decomposition: the first term is linear in the state ${\mathbf{x}}(t)$, whereas the neural term may depend nonlinearly on ${\mathbf{x}}(t)$. This separation is not unique, but it is particularly useful numerically because the known linear part can often be handled analytically, leaving only the neural contribution to be approximated numerically.

For the three standard semilinear parameterizations[^3], the corresponding choices of $a(t)$, $b(t)$, and ${\mathbf{N}}_{\boldsymbol{\phi}^\times}$ are listed in [Table 9.1](01-prologue.md#tb-pf-ode-parameterizations). For example, under noise prediction,

<span id="eq-noise-pf-ode"></span>

$$
\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t}
=
\underbrace{
f(t){\mathbf{x}}(t)
}_{\text{linear part}}
+
\underbrace{
\frac{1}{2}\frac{g^2(t)}{\sigma_t}
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}(t),t)
}_{\text{nonlinear neural part}}.
\tag{9.1.5}
$$

The coefficient multiplying the neural prediction is known entirely from the noise schedule; the only learned quantity in the second term is ${\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}(t),t)$.

<span id="tb-pf-ode-parameterizations"></span>

**Table 9.1: Factored PF-ODE forms for score, noise, and clean-data prediction.**

<table markdown="1" style="display: table; overflow: visible;">
<tr markdown="1">
<th markdown="block">

**Prediction**

</th>
<th markdown="block">

$\boldsymbol{a(t)}$

</th>
<th markdown="block">

$\boldsymbol{b(t)}$

</th>
<th markdown="block">

$\boldsymbol{{\mathbf{N}}_{\boldsymbol{\phi}^\times}}$

</th>
</tr>
<tr markdown="1">
<td markdown="block">

${\mathbf{s}}$-prediction

</td>
<td markdown="block">

$f(t)$

</td>
<td markdown="block">

$-\frac{1}{2}g^2(t)$

</td>
<td markdown="block">

${\mathbf{s}}_{\boldsymbol{\phi}^\times}$

</td>
</tr>
<tr markdown="1">
<td markdown="block">

${\boldsymbol{\epsilon}}$-prediction

</td>
<td markdown="block">

$f(t)$

</td>
<td markdown="block">

$\displaystyle \tfrac{1}{2}\tfrac{g^2(t)}{\sigma_t}$

</td>
<td markdown="block">

${\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}$

</td>
</tr>
<tr markdown="1">
<td markdown="block">

${\mathbf{x}}$-prediction

</td>
<td markdown="block">

$\displaystyle \tfrac{\sigma_t'}{\sigma_t}$

</td>
<td markdown="block">

$\displaystyle \alpha_t \left( \tfrac{\alpha_t'}{\alpha_t} - \tfrac{\sigma_t'}{\sigma_t} \right)$

</td>
<td markdown="block">

${\mathbf{x}}_{\boldsymbol{\phi}^\times}$

</td>
</tr>
</table>

At the oracle level, score, noise, clean-data, and velocity prediction are exactly equivalent representations of the same PF-ODE. For an empirical model, the same equivalence holds when the different prediction parameterizations are obtained through exact algebraic conversion of the same model output (see [Equation (6.3.1)](../06-unified-perspective/03-equivalent-parameterizations.md#eq-predictions-equivalence)). In this case, the same empirical PF-ODE can be expressed through score, noise, clean-data, or velocity prediction. The resulting formulas look different because they expose different numerical structures, but they describe the same learned continuous dynamics. However, if separate networks are trained independently for different prediction targets, their empirical PF-ODEs need not coincide exactly. In that case, differences between samplers may reflect both model approximation and numerical discretization.

**III. Exponential Integrators for Semilinear PF-ODEs.** The factored semilinear form in [Equation (9.1.4)](01-prologue.md#eq-general-semilinear-pfode) separates the analytically known linear dynamics from the model-dependent prediction. For $s>t$, define the integrating factor

$$
\mathcal{E}(s\shortrightarrow t)
:=
\exp\left(
\int_s^t a(u)\mathop{}\!\mathrm{d} u
\right).
$$

The variation-of-constants formula then gives the exact empirical PF-ODE flow

<span id="eq-factored-variation-of-constants"></span>

$$
\widetilde{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
=
\mathcal{E}(s\shortrightarrow t){\mathbf{x}}_s
+
\int_s^t
\mathcal{E}(\tau\shortrightarrow t)
b(\tau)
{\mathbf{N}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)
\mathop{}\!\mathrm{d}\tau .
\tag{9.1.6}
$$

We refer the reader to [Section A.1.3](../../appendices/a-differential-equations/01-ordinary-differential-equations.md#app-exp-int-factor) for the general derivation.

For example, under noise prediction, $a(t)=f(t)$, $b(t)=\tfrac{1}{2}g^2(t)/\sigma_t$, and ${\mathbf{N}}_{\boldsymbol{\phi}^\times}={\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}$, so

<span id="eq-variation-of-constants"></span>

$$
\widetilde{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
=
\underbrace{
\mathcal{E}(s\shortrightarrow t){\mathbf{x}}_s
}_{\text{exact linear evolution}}
+
\frac{1}{2}
\int_s^t
\frac{g^2(\tau)}{\sigma_\tau}
\mathcal{E}(\tau\shortrightarrow t)
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)
\mathop{}\!\mathrm{d}\tau,
\tag{9.1.7}
$$

where $\mathcal{E}(s\shortrightarrow t) = \exp\!\left(\int_s^t f(u)\mathop{}\!\mathrm{d} u\right)$.

The numerical opportunity is now clear. The integrating factor and the coefficient $b(\tau)$ are determined by the diffusion schedule; the only unknown quantity inside the integral is the neural prediction evaluated along the trajectory. Keeping these known factors explicit therefore allows a first-order approximation to act only on the neural prediction, instead of absorbing the coefficient-weighted product $b(\tau){\mathbf{N}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)$ into a single unknown residual.

**Why the Factored Exponential-Integrator Form Can Help.** To see the benefit of this separation, compare it with plain Euler over one backward step from an exact empirical PF-ODE state ${\mathbf{x}}_s$ at time $s$ to $t<s$. Define $\Delta s:=s-t>0$ and ${\mathbf{N}}_s:={\mathbf{N}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_s,s)$.

Plain Euler treats the complete PF-ODE velocity as locally constant, $a(\tau){\mathbf{x}}_\tau+ b(\tau){\mathbf{N}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau) \approx a(s){\mathbf{x}}_s+b(s){\mathbf{N}}_s$ for $\tau\in[t,s]$, giving

<span id="eq-euler-step"></span>

$$
\begin{aligned}
\tilde{\mathbf{x}}_t^{\mathrm{Euler}}
=
{\mathbf{x}}_s
-
\Delta s
\left[
a(s){\mathbf{x}}_s+b(s){\mathbf{N}}_s
\right].
\end{aligned}
\tag{9.1.8}
$$

Instead, we may hold only the neural prediction fixed, ${\mathbf{N}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)\approx{\mathbf{N}}_s$, while retaining the known schedule dependence in [Equation (9.1.6)](01-prologue.md#eq-factored-variation-of-constants). This gives the *first-order factored exponential-integrator update*[^4]

<span id="eq-factored-ei-step"></span>

$$
\begin{aligned}
\tilde{\mathbf{x}}_t^{\mathrm{EI}}
=
\mathcal E(s\shortrightarrow t){\mathbf{x}}_s
+
\left[
\int_s^t
\mathcal E(\tau\shortrightarrow t)b(\tau)\mathop{}\!\mathrm{d}\tau
\right]{\mathbf{N}}_s.
\end{aligned}
\tag{9.1.9}
$$

Both methods use one neural-network evaluation per step. Plain Euler freezes the complete PF-ODE velocity, while the factored update keeps the known schedule dependence and freezes only the neural prediction. This factorization matters: if $b(\tau){\mathbf{N}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)$ is instead treated as a single residual and held fixed, the result is conventional exponential Euler. Thus, the factored update retains one more known quantity, $b(\tau)$, inside the integral; the two updates coincide when $b(\tau)$ is constant over the step.

A simple frozen-prediction example makes this difference concrete. Suppose $a(t)\equiv a$, $b(t)\equiv b$, and ${\mathbf{N}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)\equiv{\mathbf{N}}_s$ over the step. For $a\neq0$,

$$
\begin{aligned}
\tilde{\mathbf{x}}_t^{\mathrm{EI}}
&=
e^{-a\Delta s}{\mathbf{x}}_s
+
\frac{b}{a}
\left(
e^{-a\Delta s}-1
\right){\mathbf{N}}_s,
\\
\tilde{\mathbf{x}}_t^{\mathrm{Euler}}
&=
(1-a\Delta s){\mathbf{x}}_s
-
b\Delta s\,{\mathbf{N}}_s.
\end{aligned}
$$

Expanding the factored update gives

$$
\begin{aligned}
\tilde{\mathbf{x}}_t^{\mathrm{EI}}
=
\tilde{\mathbf{x}}_t^{\mathrm{Euler}}
+
\frac{(\Delta s)^2}{2}
\left[
a^2{\mathbf{x}}_s
+
ab\,{\mathbf{N}}_s
\right]
+
\mathcal O((\Delta s)^3).
\end{aligned}
$$

Thus, the two updates agree to first order for small steps, while the factored update retains the higher-order effects of the known dynamics. The difference can become important for larger steps. For example, when $a>0$, the exact multiplier $e^{-a\Delta s}$ remains positive and contractive for every $\Delta s>0$, whereas the Euler approximation $1-a\Delta s$ becomes negative when $a\Delta s>1$ and has magnitude greater than one when $a\Delta s>2$. Plain Euler can therefore distort dynamics that the factored form already handles analytically.

In the full PF-ODE, the neural prediction generally varies along the trajectory. Subtracting [Equation (9.1.9)](01-prologue.md#eq-factored-ei-step) from the exact variation-of-constants formula gives

<span id="eq-ei-local-defect"></span>

$$
\begin{aligned}
\widetilde{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
-
\tilde{\mathbf{x}}_t^{\mathrm{EI}}
=
\int_s^t
\mathcal E(\tau\shortrightarrow t)b(\tau)
\Big[
{\mathbf{N}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau,\tau)
-
{\mathbf{N}}_s
\Big]\mathop{}\!\mathrm{d}\tau.
\end{aligned}
\tag{9.1.10}
$$

Hence, once the known schedule-dependent dynamics are handled analytically, the remaining discretization error is governed by the variation of the neural prediction along the trajectory. This separation is the basic numerical principle that several diffusion-specific solvers will exploit in different ways.

## 9.1.3 Roadmap from Numerical Ideas to Diffusion Samplers {#roadmap}

The remainder of this chapter shows how these numerical ideas appear in modern diffusion samplers. We begin with DDIM in [Section 9.2](02-ddim.md#sec-revisit-ddim), which constructs a first-order update from the current model prediction. We then study DEIS in [Section 9.3](03-deis.md#sec-exponential), which uses predictions from previous steps to build higher-order approximations. DPM-Solver in [Section 9.4](04-dpm-solver.md#sec-dpm) develops higher-order updates using additional model evaluations within the current step. DPM-Solver`++` in [Section 9.5](05-dpm-solver-plus-plus.md#sec-dpm) develops two corresponding variants based on clean-data prediction: a single-step variant, DPM-Solver`++`(2S), which uses additional evaluations within the current step, and a multistep variant, DPM-Solver`++`(2M), which reuses predictions from previous steps. Finally, [Section 9.6](06-paradigms.md#sec-picard) introduces ParaDiGMS, which updates several time points together through fixed-point iteration to enable parallel computation across time.

The table below provides a roadmap connecting these diffusion samplers with the classical numerical ideas reviewed in [Equation (A.1.6)](../../appendices/a-differential-equations/01-ordinary-differential-equations.md#eq-classic-solvers).

<table markdown="1" style="display: table; overflow: visible;">
<tr markdown="1">
<th markdown="block">

**Classical Idea**

</th>
<th markdown="block">

**How the Update Is Constructed**

</th>
<th markdown="block">

**Diffusion Examples**

</th>
<th markdown="block">

**Where**

</th>
</tr>
<tr markdown="1">
<td markdown="block">

Euler-type first-order update

</td>
<td markdown="block">

Use the model prediction at the beginning of the step to advance to the next state.

</td>
<td markdown="block">

DDIM; DPM-Solver-1

</td>
<td markdown="block">

[Section 9.2](02-ddim.md#sec-revisit-ddim) and [Section 9.4](04-dpm-solver.md#sec-dpm)

</td>
</tr>
<tr markdown="1">
<td markdown="block">

RK-type staged update

</td>
<td markdown="block">

Evaluate the model at additional intermediate states before completing the current step.

</td>
<td markdown="block">

Higher-order DPM-Solver; DPM-Solver`++`(2S)

</td>
<td markdown="block">

[Section 9.4](04-dpm-solver.md#sec-dpm) and [Section 9.5](05-dpm-solver-plus-plus.md#sec-dpm)

</td>
</tr>
<tr markdown="1">
<td markdown="block">

AB-type multistep update

</td>
<td markdown="block">

Reuse model predictions from previous steps to construct the next update.

</td>
<td markdown="block">

DEIS; DPM-Solver`++`(2M)

</td>
<td markdown="block">

[Section 9.3](03-deis.md#sec-exponential) and [Section 9.5](05-dpm-solver-plus-plus.md#sec-dpm)

</td>
</tr>
<tr markdown="1">
<td markdown="block">

Picard-type fixed-point iteration

</td>
<td markdown="block">

Update several time points together and repeatedly refine their states.

</td>
<td markdown="block">

ParaDiGMS

</td>
<td markdown="block">

[Section 9.6](06-paradigms.md#sec-picard)

</td>
</tr>
</table>

These connections provide a high-level numerical roadmap for the methods developed below. Diffusion solvers further benefit from structure specific to the PF-ODE, including different prediction parameterizations, analytically known schedule-dependent coefficients, and alternative time parameterizations.

To keep the presentation systematic, each section follows the same broad structure: we first derive the diffusion solver and explain the numerical idea it exploits, then connect it to the corresponding classical solver principle, and finally relate it to other diffusion solvers developed in this chapter. This organization makes clear both the common numerical backbone and the diffusion-specific ingredients that distinguish the resulting methods.

[^1]: To see the scaling, recall the oracle relation from [Equation (6.3.1)](../06-unified-perspective/03-equivalent-parameterizations.md#eq-predictions-equivalence):

    $$
    {\boldsymbol{\epsilon}}^*({\mathbf{x}}_t,t)
        =
        -\sigma_t\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}}_t),
    $$

    where ${\boldsymbol{\epsilon}}^*({\mathbf{x}}_t,t) = \mathbb{E}[{\boldsymbol{\epsilon}}|{\mathbf{x}}_t]$ is the oracle noise predictor, and the perturbation kernel is ${\mathbf{x}}_t|{\mathbf{x}}_0 \sim \mathcal{N}({\mathbf{x}}_t; \alpha_t{\mathbf{x}}_0,\sigma_t^2{\mathbf{I}})$. By the orthogonality property of conditional expectation in $L^2$,

    $$
    \mathbb{E}\|{\boldsymbol{\epsilon}}\|_2^2
        =
        \mathbb{E}\|{\boldsymbol{\epsilon}}^*\|_2^2
        +
        \mathbb{E}\|{\boldsymbol{\epsilon}}-{\boldsymbol{\epsilon}}^*\|_2^2,
    $$

    and therefore

    $$
    \mathbb{E}\|{\boldsymbol{\epsilon}}^*\|_2^2
        \le
        \mathbb{E}\|{\boldsymbol{\epsilon}}\|_2^2
        =
        D,
    $$

    where $D$ is the data dimension. Thus, the oracle noise predictor has uniformly bounded expected squared norm. In contrast, the oracle score

    $$
    {\mathbf{s}}^*({\mathbf{x}}_t,t)
        :=
        \nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}}_t)
    $$

    satisfies

    $$
    \mathbb{E}\|{\mathbf{s}}^*({\mathbf{x}}_t,t)\|_2^2
        =
        \sigma_t^{-2}\,
        \mathbb{E}\|{\boldsymbol{\epsilon}}^*({\mathbf{x}}_t,t)\|_2^2
        \le
        \frac{D}{\sigma_t^2}.
    $$

    Thus, direct score prediction can become increasingly poorly scaled as $\sigma_t$ becomes small, since noise prediction is converted to the score through the factor $1/\sigma_t$. This scaling can make direct score parameterization numerically less favorable near small noise levels.

[^2]: In standard semilinear notation, $b(t){\mathbf{N}}_{\boldsymbol{\phi}^\times}$ could be absorbed into a single nonlinear term. We keep the two factors separate because $b(t)$ is known analytically from the noise schedule, whereas ${\mathbf{N}}_{\boldsymbol{\phi}^\times}$ requires a neural-network evaluation. This allows a solver to approximate the neural prediction inside the integral while retaining and analytically integrating the known schedule-dependent coefficient.

[^3]: In the following sections, we switch between the SDE coefficients $(f(t),g(t))$ and the Gaussian perturbation kernel ${\mathbf{x}}_t | {\mathbf{x}}_0 \sim \mathcal{N} \bigl( {\mathbf{x}}_t; \alpha_t{\mathbf{x}}_0, \sigma_t^2{\mathbf{I}} \bigr)$. They are related by

    $$
    f(t)
        =
        \frac{\alpha_t'}{\alpha_t},
        \qquad
        g^2(t)
        =
        \frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}\bigl(\sigma_t^2\bigr)
        -
        2\frac{\alpha_t'}{\alpha_t}\sigma_t^2
        =
        2\sigma_t\sigma_t'
        -
        2\frac{\alpha_t'}{\alpha_t}\sigma_t^2.
    $$

[^4]: The update in [Equation (9.1.9)](01-prologue.md#eq-factored-ei-step) is not plain Euler in the original state and time variables. After an integrating-factor transformation of the state and, on intervals where the resulting time coordinate is monotone, a corresponding time reparameterization, it can be written as an ordinary explicit Euler step. We make this connection concrete for diffusion sampling in the DDIM section.
