# 10.1 Prologue {#sec-distillation-prologue}

A central bottleneck of diffusion models is their slow sampling speed.

As shown through Tweedie’s formula ([Section 6.3.1](../06-unified-perspective/03-equivalent-parameterizations.md#subsec-four-equiv-para)), a diffusion model can be interpreted as an “${\mathbf{x}}$-prediction” model, ${\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t, t)$, trained to recover the expected clean data from a noisy input ${\mathbf{x}}_t$ at noise level $t$:

$$
{\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t, t) \approx \mathbb{E}[{\mathbf{x}}_0 |{\mathbf{x}}_t],
$$

where the expectation is taken with respect to $p({\mathbf{x}}_0 |{\mathbf{x}}_t)$, representing all plausible clean data corresponding to ${\mathbf{x}}_t$. A natural idea is to use ${\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t, t)$ for one step generation. Yet because this denoiser averages over many plausible outcomes, the prediction can become overly smooth, and generation with only a few denoising steps leads to blurry, low quality samples.

On the other hand, as discussed in [Section 4.3.2](../04-score-sde-framework/03-training-and-sampling.md#subsec-sde-sample), diffusion sampling follows an ODE or SDE trajectory through a long sequence of iterative steps. This produces high fidelity samples, but repeated neural-network evaluations make sampling expensive. For a deterministic ODE solver of order $n$, under standard regularity assumptions, the local truncation error is $\mathcal{O}(h^{n+1})$ and the global discretization error over a fixed time interval is $\mathcal{O}(h^n)$, where $h=\max_i |t_i-t_{i-1}|$ is the step size. Thus, using a coarser discretization generally increases numerical error for a fixed learned vector field, although sample fidelity need not decrease monotonically with the number of solver steps. This creates a practical trade off between sampling accuracy and computational cost.

To overcome this bottleneck, a major line of research is *distillation*, which assumes access to a well-trained diffusion model (the *teacher*) and trains a generator (the *student*) to reproduce its behavior through a single feed forward or few step computation. This compresses the teacher’s many sampling steps into a fast process, effectively bypassing slow iterative solvers while maintaining high sample fidelity.

Below, we introduce two perspectives on distillation: *distribution level distillation* and *flow map level distillation*[^1].

## 10.1.1 Distribution Level Distillation {#distribution-level-distillation}

The goal of distribution-based distillation is to train a one-step generator ${\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{z}})$ that maps noise ${\mathbf{z}} \sim p_{\mathrm{prior}}$ to a sample $\hat{\mathbf{x}} = {\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{z}})$, inducing a distribution $p_{\boldsymbol{\theta}}(\hat{\mathbf{x}})$ that approximates the target data distribution $p_{\mathrm{data}}({\mathbf{x}})$. This is typically achieved by minimizing a statistical divergence

$$
\min_{\boldsymbol{\theta}}  \mathcal{D}\!\left(p_{\boldsymbol{\theta}}(\hat{\mathbf{x}}),  p_{\mathrm{data}}(\hat{\mathbf{x}})\right),
$$

where $\mathcal{D}$ denotes a suitable divergence measurement such as KL.

In practice, distribution based methods align the generator’s distribution with the empirical distribution $p_{\boldsymbol{\phi}^\times}({\mathbf{x}})$ produced by a pre-trained diffusion model:

$$
\min_{\boldsymbol{\theta}}  \mathcal{D}\!\left(p_{\boldsymbol{\theta}}(\hat{\mathbf{x}}),  p_{\boldsymbol{\phi}^\times}(\hat{\mathbf{x}})\right),
$$

where $p_{\boldsymbol{\phi}^\times}$ serves as a surrogate for $p_{\mathrm{data}}$. Rather than evaluating this divergence explicitly, these methods construct tractable estimators of its gradient using the pre-trained teacher together with quantities estimated from the student’s current distribution. For KL-based distribution matching, this gradient takes the form of a difference between the student and target score functions.

This formulation distills multi-step generative processes of diffusion models into a single step model through distributional alignment. We detail this approach in [Section 10.2](02-distribution-based-distillation.md#sec-vsd).

## 10.1.2 Flow Map Level Distillation {#flow-map-level-distillation}

We consider the PF-ODE, which can be expressed for any prediction model (see [Equation (6.3.1)](../06-unified-perspective/03-equivalent-parameterizations.md#eq-predictions-equivalence)):

<span id="eq-pf-ode-ct"></span>

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(\tau)}{\mathop{}\!\mathrm{d} \tau}
    = f(\tau) {\mathbf{x}}(\tau) - \tfrac{1}{2}g^2(\tau) \nabla_{\mathbf{x}} \log p_\tau({\mathbf{x}}(\tau))
    =: {\mathbf{v}}^*({\mathbf{x}}(\tau), \tau).
\end{aligned}
\tag{10.1.1}
$$

Its solution map (flow map), starting from ${\mathbf{x}}_s$ at time $s$ and evolving reversely to time $t \leq s$, is denoted by ${\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)$; that is,

<span id="eq-int-target-gt"></span>

$$
\begin{aligned}
    {\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
    &\coloneqq {\mathbf{x}}_s + \int_s^t {\mathbf{v}}^*({\mathbf{x}}(\tau), \tau) \mathop{}\!\mathrm{d} \tau,
\end{aligned}
\tag{10.1.2}
$$

where the integral solves the PF-ODE. Intuitively, ${\boldsymbol{\Psi}}_{s\to t}$ transports, ${\mathbf{x}}_s$, noise at time $s$ to less noisy states at time $t$ (ultimately data at $t=0$).

Sampling from a diffusion model corresponds to evaluating ${\boldsymbol{\Psi}}_{T\to 0}({\mathbf{x}}_T)$ for ${\mathbf{x}}_T \sim p_{\mathrm{prior}}$. Typically, this integral is approximated by iterative numerical solvers leveraging the velocity field ${\mathbf{v}}$ (see [Chapter 9](../09-numerical-solvers/index.md#ch-numerical-solvers)), but requires many steps (e.g., at least $10$-$20$ NFEs even in DPM-Solver), making sampling slower than classic one-step generative models such as GAN. This motivates a natural question:

<span id="question-10-1-1"></span>

**Question 10.1.1**

Can we learn the solution map ${\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)$ directly?

In particular, learning a map ${\boldsymbol{\Psi}}_{T\to 0}({\mathbf{x}}_T)$ with ${\mathbf{x}}_T \sim p_{\mathrm{prior}}$ enables one-step generation.

**Trajectory Distillation.**

Trajectory distillation seeks to train a neural generator that approximates the solution map at the instance level. Since the PF-ODE integral rarely admits a closed form, it must be approximated numerically during training. To formalize, we introduce the general solver notation

<span id="eq-solver"></span>

$$
\begin{aligned}
\mathtt{Solver}_{s\to t}({\mathbf{x}}_s;\,\boldsymbol{\phi}^\times)
    \quad \text{or simply} \quad
    \mathtt{Solver}_{s\to t}({\mathbf{x}}_s),
\end{aligned}
\tag{10.1.3}
$$

denoting numerical integration of the empirical PF-ODE from $s$ to $t$ starting at ${\mathbf{x}}_s$, with teacher parameters $\boldsymbol{\phi}^\times$ (omitted when clear from context).

**An Early Approach: Direct Knowledge Distillation.** To enable few step or even one step generation, a direct approach is to train a generator ${\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{x}}_T, T,0)$ to imitate the output of a numerical solver evaluated along the full trajectory:

$$
{\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{x}}_T, T,0)\approx \mathtt{Solver}_{T\to 0}({\mathbf{x}}_T),
\qquad {\mathbf{x}}_T \sim p_{\mathrm{prior}}.
$$

This idea underlies one of the earliest trajectory distillation methods, *Knowledge Distillation* ([Luhman and Luhman, 2021](../../references.md#ref-luhman2021knowledge)), which uses the regression loss

$$
\mathcal{L}_{\mathrm{KD}}({\boldsymbol{\theta}})
:=
{\mathbb{E}}_{{\mathbf{x}}_T \sim p_{\mathrm{prior}}}
\left\|
{\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{x}}_T, T,0)
-
\mathtt{Solver}_{T\to 0}({\mathbf{x}}_T)
\right\|_2^2.
$$

While this approach provides direct supervision from the pre-trained teacher, it cannot leverage the strong supervision available in the original training data. In addition, it is computationally expensive if ODE integration is invoked within the training loop, since each parameter update requires solving the ODE to form targets. Finally, because the generator learns only a global mapping from $T$ to $0$, it may lose controllability for steering the generation process from intermediate states. Consequently, most controllable generation techniques introduced in [Chapter 8](../08-guidance/index.md#ch-guidance) cannot be directly applied.

**Preface to Progressive Distillation.** *Progressive Distillation* (PD) ([Salimans and Ho, 2021](../../references.md#ref-salimans2021progressive)) trains a time–conditional $\mathtt{Student}$ using *local* supervision from $\mathtt{Teacher}$ fragments. Let $t_0=T>t_1>\cdots>t_N=0$ be a fixed time grid. The $\mathtt{Teacher}$ provides time–stepping maps $\mathtt{Teacher}_{t_k\to t_{k+1}}$ for $k=0,\ldots,N-1$.

Rather than supervising only the one-jump $T\to 0$, PD trains the $\mathtt{Student}$ two–step skip map to match two consecutive $\mathtt{Teacher}$ steps:

$$
\mathtt{Student}_{t_k\to t_{k+2}}
\;\approx\;
\mathtt{Teacher}_{t_{k+1}\to t_{k+2}}
\circ
\mathtt{Teacher}_{t_{k}\to t_{k+1}},
$$

for $k = 0, 2, 4, \dots$. The matching is performed using a simple regression loss (e.g., mean squared error).

After training on locally paired fragments, the $\mathtt{Student}$ no longer follows every time interval of the original grid. Instead, it advances on every other time point,

$$
t_0 \to t_2 \to t_4 \to \cdots \to t_N,
$$

which means that each $\mathtt{Student}$ step effectively covers two consecutive $\mathtt{Teacher}$ steps. Consequently, the $\mathtt{Student}$ completes the same overall time span $[0,T]$ using only $N/2$ transitions.

After this stage, the trained $\mathtt{Student}$ replaces the $\mathtt{Teacher}$ to serve as the new reference model. The entire procedure is then repeated on the coarser grid, where the time step doubles $(N \to N/2 \to N/4 \to \cdots)$, progressively distilling the trajectory into fewer and fewer steps until the desired number of inference steps is reached. This iterative halving preserves the global time horizon while continually compressing the temporal resolution of the generative process.

**A Unified Perspective of Flow Map Learning.** Various methods, including KD and PD, can be expressed within a unified loss framework:

<span id="eq-general-cm-loss"></span>

$$
\begin{aligned}
\mathcal{L}_{\mathrm{oracle}}({\boldsymbol{\theta}})
  := {\mathbb{E}}_{s,t} {\mathbb{E}}_{{\mathbf{x}}_s \sim p_s}
  \left[  w(s,t)  d \big({\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{x}}_s, s,t),  {\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)\big)\right],
\end{aligned}
\tag{10.1.4}
$$

where ${\boldsymbol{\Psi}}_{s\to t}$ is the oracle flow map, $w(s,t) \geq 0$ specifies how different time pairs $(s,t)$ are weighted, $d(\cdot,\cdot)$ is a discrepancy measure such as $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|_2^2$ or $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|_1$, and $p_s$ denotes the forward noised marginal at time $s$. Because ${\boldsymbol{\Psi}}_{s\to t}$ is not available in closed form, one must rely on approximations, typically through a pre-trained diffusion model (teacher) or another tractable surrogate.

KD appears as a simple instance of [Equation (10.1.4)](01-prologue.md#eq-general-cm-loss) by considering only the endpoint pair $(s,t)=(T,0)$. Equivalently, one may take the degenerate weighting $w(s,t)=\delta(s{-}T)\delta(t)$. Assuming $p_T=p_{\mathrm{prior}}$[^2], the oracle objective becomes

$$
{\mathbb{E}}_{{\mathbf{x}}_T\sim p_T}
\big\|{\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{x}}_T,T,0)-{\boldsymbol{\Psi}}_{T\to 0}({\mathbf{x}}_T)\big\|_2^2
\approx \mathcal{L}_{\mathrm{KD}}({\boldsymbol{\theta}}),
$$

where the practical KD target replaces the oracle flow map by $\mathtt{Solver}_{T\to 0}\approx{\boldsymbol{\Psi}}_{T\to 0}$. An alternative perspective on this formulation is presented in [Section D.5](../../appendices/d-supplementary-proofs/05-fast-generators.md#app-sec-flow-map).

PD also fits this template, but instead of supervising only with the single extreme pair $(T,0)$, it uses many *nearby* time pairs and enforces a simple *local consistency* rule: a short step followed by another short step should match the direct two–step move. We return to this in [Equation (10.3.3)](03-progressive-distillation.md#eq-pd-local).

In practice, the main challenge is that the oracle flow map ${\boldsymbol{\Psi}}_{s\to t}$ generally has no closed-form expression, making direct supervision infeasible. A range of methods have been developed to approximate this target efficiently, but their success often hinges on the quality of the teacher model. We will return to [Equation (10.1.4)](01-prologue.md#eq-general-cm-loss) in [Chapter 11](../11-fast-generators/index.md#ch-fast-scratch), presenting a principled framework for training-from-scratch methods that eliminate the teacher from the learning loop.

[^1]: Chronologically, flow map level distillation appeared earlier, with *Knowledge Distillation* (KD) ([Luhman and Luhman, 2021](../../references.md#ref-luhman2021knowledge)) in 2021 and *Progressive Distillation* (PD) ([Salimans and Ho, 2021](../../references.md#ref-salimans2021progressive)) in 2022, preceding the family of distribution level distillation approaches that emerged around 2023. For smoother exposition and connection to the next chapter, however, we present distribution level distillation first.

[^2]: This equality holds when the noise schedule reaches the prior exactly at $T$; otherwise, one typically has $p_T\approx p_{\mathrm{prior}}$ for sufficiently large $T$.
