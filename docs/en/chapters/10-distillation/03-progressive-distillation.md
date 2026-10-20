# 10.3 Progressive Distillation {#sec-pd}

Progressive Distillation (PD) ([Salimans and Ho, 2021](../../references.md#ref-salimans2021progressive)) consists of two procedures that together enable a diffusion model to learn the PF-ODE trajectory more efficiently. The key idea is to progressively reduce the number of integration steps required for high-quality sampling while retaining fidelity to the teacher trajectory.

-   **Distillation Operation:** Distills a deterministic sampler (e.g., DDIM) based on a pre-trained teacher model (initially a diffusion model) into a student model that reproduces the same trajectory using only half as many sampling steps.

-   **Progressive Operation:** Repeats this distillation process iteratively, each time halving the number of steps, until the student can generate high-quality samples within a small fixed budget (typically $1$–$4$ steps).

<span id="fig-pd-illustration"></span>

![Progressive distillation compresses two teacher steps into one student step](../../assets/chapter-10/progressive-distillation.png)

**Figure 10.1:** **Illustration of Progressive Distillation (PD).** At each round, the student model is trained so that a single step reproduces the effect of two adjacent teacher steps. This process distills $N$ teacher steps into $N/2$ student steps, and repeating the procedure progressively halves the trajectory length until the desired step count is reached. The arrows indicate how multi-step teacher transitions are compressed into fewer student steps, moving from noise to data.

Source: Created by the authors.

We first introduce the distillation operation of PD in [Section 10.3.1](03-progressive-distillation.md#subsec-pd-distillation), and then summarize the entire training pipeline in [Section 10.3.2](03-progressive-distillation.md#subsec-pd-training). [Section 10.3.4](03-progressive-distillation.md#subsec-pd-guidance) presents an extension for CFG guidance.

## 10.3.1 Distillation Operation in PD {#subsec-pd-distillation}

In this section, we fix DDIM in the ${\mathbf{x}}$-prediction parameterization as the time–stepping rule and still write $\mathtt{Solver}_{s \to t}$ for the deterministic map obtained by plugging the current teacher’s ${\mathbf{x}}$-denoiser into DDIM.

In the first PD round (teacher = pre–trained diffusion model), this coincides with integrating the diffusion PF–ODE via DDIM; in later rounds (teacher = previous student), $\mathtt{Solver}_{s \to t}$ is simply the DDIM transition induced by the current teacher, not the original diffusion PF–ODE.

The distillation step is as follows: starting from a noisy input ${\mathbf{x}}_s$ (a perturbed version of clean data, ${\mathbf{x}}_s = \alpha_s {\mathbf{x}}_0 + \sigma_s \boldsymbol{\epsilon}$), the student is trained to predict a target ${\color{orange}\tilde{\mathbf{x}}}$ so that a single student step $s\!\to\! t$ reproduces the teacher’s two consecutive steps $s\!\to\! u\!\to\! t$. Let ${\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}},\tau)$ denote the teacher’s ${\mathbf{x}}$-prediction denoiser in this round. Applying the teacher–induced DDIM transition twice gives

$$
\tilde{\mathbf{x}}_u := \mathtt{Solver}_{s\to u}\!\left({\mathbf{x}}_s;\,{\mathbf{x}}_{\boldsymbol{\phi}^\times}\right),
\qquad
\tilde{\mathbf{x}}_t := \mathtt{Solver}_{u\to t}\!\left(\tilde{\mathbf{x}}_u;\,{\mathbf{x}}_{\boldsymbol{\phi}^\times}\right).
$$

Here, we use the notation of [Equation (10.1.3)](01-prologue.md#eq-solver) to denote the deterministic transition map from $s$ to $t$ (starting at ${\mathbf{x}}_s$) induced by plugging ${\mathbf{x}}_{\boldsymbol{\phi}^\times}$ into DDIM.

<span id="question-10-3-1"></span>

**Question 10.3.1**

What is the pseudo-clean ${\color{orange}\tilde{\mathbf{x}}}$ at time $s$ such that the solver produces the same output $\tilde{\mathbf{x}}_t$ when stepping directly $s \to t$ as it does via $s \to u \to t$? Specifically, determine ${\color{orange}\tilde{\mathbf{x}}}$ satisfying:

$$
\tilde{\mathbf{x}}_t = \mathtt{Solver}_{s\to t}\left({\mathbf{x}}_s; {\color{orange}\tilde{\mathbf{x}}}\right).
$$

Once a closed-form expression for ${\color{orange}\tilde{\mathbf{x}}}$ is obtained, we train a student model ${\mathbf{f}}_{\boldsymbol{\theta}}({\mathbf{x}}_s, s)$ (also an ${\mathbf{x}}$-prediction model here) to approximate the “two-steps-in-one” target ${\color{orange}\tilde{\mathbf{x}}}$ by minimizing

<span id="eq-pd-optimization"></span>

$$
\begin{aligned}
\min_{\boldsymbol{\theta}} {\mathbb{E}}_s
\mathbb{E}_{{\mathbf{x}}_s\sim p_s}\!\left[w(\lambda_s)\,\big\|{\mathbf{f}}_{\boldsymbol{\theta}}({\mathbf{x}}_s, s) - {\color{orange}\tilde{\mathbf{x}}}\big\|_2^2\right].
\end{aligned}
\tag{10.3.1}
$$

In the following, we show that the DDIM rule yields ${\color{orange}\tilde{\mathbf{x}}}$ in closed form through elementary algebra (note that the result holds for both discrete and continuous time):

<span id="pd-ddim"></span>

**Lemma 10.3.1: Two-Steps-in-One Target ${\color{orange}\tilde{\mathbf{x}}}$ of DDIM**

Starting from an initial condition ${\mathbf{x}}_s$, if the solver is taken as DDIM, then the “two-step-in-one” target ${\color{orange}\tilde{\mathbf{x}}}$ can be computed as

$$
\begin{aligned}
{\color{orange}\tilde{\mathbf{x}}}
    &=\frac{\sigma_s}{\alpha_t\sigma_s - \alpha_s\sigma_t} \tilde{\mathbf{x}}_t - \frac{\sigma_t}{\alpha_t\sigma_s - \alpha_s\sigma_t} {\mathbf{x}}_s.
\end{aligned}
$$

Here, $\tilde{\mathbf{x}}_t$ is obtained by applying DDIM (in [Equation (9.2.2)](../09-numerical-solvers/02-ddim.md#eq-ddim-update-all)) twice, from $s \to u \to t$:

$$
\begin{aligned}
s \to u: & \quad    && \tilde{\mathbf{x}}_u = \frac{\sigma_u}{\sigma_s}{\mathbf{x}}_s + \alpha_s \left(\frac{\alpha_u}{\alpha_s} - \frac{\sigma_u}{\sigma_s} \right){\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_s, s)  \\
    u \to t: & \quad    && \tilde{\mathbf{x}}_t = \frac{\sigma_t}{\sigma_u}\tilde{\mathbf{x}}_u + \alpha_u \left(\frac{\alpha_t}{\alpha_u} - \frac{\sigma_t}{\sigma_u} \right){\mathbf{x}}_{\boldsymbol{\phi}^\times}(\tilde{\mathbf{x}}_u, u).
\end{aligned}
$$

**Proof for Lemma.**

$\tilde{\mathbf{x}}_t$ must be matched with the one-step DDIM from $s$ to $t$, $\tilde{\mathbf{x}}_t'$, expressed as:

$$
s \to t: \quad \tilde{\mathbf{x}}_t' = \frac{\sigma_t}{\sigma_s}{\mathbf{x}}_s + \alpha_s \left(\frac{\alpha_t}{\alpha_s} - \frac{\sigma_t}{\sigma_s} \right) {\color{orange}\tilde{\mathbf{x}}}.
$$

By equating $\tilde{\mathbf{x}}_t'$ and $\tilde{\mathbf{x}}_t$, we can solve for ${\color{orange}\tilde{\mathbf{x}}}$ in terms of $\tilde{\mathbf{x}}_t$, $s$, and $t$:

<span id="eq-x-pd-paper"></span>

$$
\begin{aligned}
&\tilde{\mathbf{x}}_t = \tilde{\mathbf{x}}_t'  \\
    \Longleftrightarrow\,\, & \tilde{\mathbf{x}}_t = \frac{\sigma_t}{\sigma_s}{\mathbf{x}}_s + \alpha_s \left(\frac{\alpha_t}{\alpha_s} - \frac{\sigma_t}{\sigma_s} \right) {\color{orange}\tilde{\mathbf{x}}}  \\
    \Longleftrightarrow\,\, & {\color{orange}\tilde{\mathbf{x}}} =  \frac{\tilde{\mathbf{x}}_t - \frac{\sigma_t}{\sigma_s}{\mathbf{x}}_s}{\alpha_s \left(\frac{\alpha_t}{\alpha_s} - \frac{\sigma_t}{\sigma_s} \right)}  \\
    \Longleftrightarrow\,\, & {\color{orange}\tilde{\mathbf{x}}} = \frac{\sigma_s}{\alpha_t\sigma_s - \alpha_s\sigma_t} \tilde{\mathbf{x}}_t - \frac{\sigma_t}{\alpha_t\sigma_s - \alpha_s\sigma_t} {\mathbf{x}}_s.
\end{aligned}
\tag{10.3.2}
$$

With this formula, PD computes the pseudo-clean target at time $s$ whose single DDIM step $s\!\to\!t$ lands exactly at the two-step output $\tilde{\mathbf{x}}_t$.

**Practical Discrete Time Grids and Loss.** In practice, we fix a decreasing grid $t_0=T>t_1>\cdots>t_N=0$ and, for brevity, write $s:=t_k$, $u:=t_{k+1}$, $t:=t_{k+2}$. The teacher provides one step maps $\mathtt{Teacher}_{t_k\to t_{k+1}}$, and the student learns a two step skip map that matches the teacher composition:

$$
\mathtt{Student}_{t_k\to t_{k+2}}
 \approx
\mathtt{Teacher}_{t_{k+1}\to t_{k+2}}
\circ
\mathtt{Teacher}_{t_{k}\to t_{k+1}}.
$$

For the standard halving schedule, assume $N$ is even and sample triplets $(s,u,t)=(t_k,t_{k+1},t_{k+2})$ with $k\in\{0,2,\ldots,N-2\}$. The objective [Equation (10.3.1)](03-progressive-distillation.md#eq-pd-optimization) becomes

$$
\min_{\boldsymbol{\theta}}
\mathbb{E}_{\,k \sim \mathcal{U}\{0,2,\ldots,N{-}2\}}
\mathbb{E}_{\,{\mathbf{x}}_{t_k}\sim p_{t_k}}\!
\Big[
  w(\lambda_{t_k})\,
  \big\|
  {\mathbf{f}}_{\boldsymbol{\theta}}({\mathbf{x}}_{t_k},t_k) - {\color{orange}\tilde{\mathbf{x}}}^{(k)}
  \big\|_2^2
\Big],
$$

where the teacher two-step target ${\color{orange}\tilde{\mathbf{x}}}^{(k)}$ is computed via [Lemma 10.3.1](03-progressive-distillation.md#pd-ddim). If the grid is uniform, one may write $t_k = T(1 - k/N)$ so that

$$
s = T\Big(1 - \frac{k}{N}\Big),\quad
u = T\Big(1 - \frac{k+1}{N}\Big),\quad
t = T\Big(1 - \frac{k+2}{N}\Big),
$$

corresponding to evenly spaced time steps of size $\Delta s = T/N$.

## 10.3.2 Entire Training Pipeline of PD and Its Sampling {#subsec-pd-training}

After training on locally paired fragments via [Equation (10.3.1)](03-progressive-distillation.md#eq-pd-optimization), the $\mathtt{Student}$ no longer follows every interval of the original grid. Instead, each learned step covers two consecutive $\mathtt{Teacher}$ steps, so the $\mathtt{Student}$ advances on every other time point,

$$
t_0 \to t_2 \to t_4 \to \cdots \to t_N,
$$

and thus traverses the same horizon $[0,T]$ using only $N/2$ transitions. After this stage, the trained $\mathtt{Student}$ replaces the $\mathtt{Teacher}$ as the new denoiser model. The procedure is then repeated on the coarser grid (the time step doubles), yielding the progression

$$
N \;\to\; N/2 \;\to\; N/4 \;\to\; \cdots,
$$

until the desired number of inference steps is reached. At each iteration, the new $\mathtt{Student}$ is initialized from the updated $\mathtt{Teacher}$. This iterative halving preserves the global time horizon while progressively compressing the temporal resolution of the generative process.

**Sampling.** At inference time, using the (DDIM) solver with the current $\mathtt{Student}$ as the denoiser, the sampler advances on the coarser grid induced by training. After the first round it takes “skip-2” jumps ($t_0 \to t_2 \to \cdots \to t_N$), after the next round “skip-4” ($t_0 \to t_4 \to \cdots \to t_N$), and so on, halving the number of sampling steps at each iteration while keeping the same start and end times.

## 10.3.3 Additional Discussion: Local Semigroup Matching and the Possibility of Generalized Solvers {#local-semigroup-matching}

**Progressive Distillation as Local Semigroup Matching.** Within the unified objective [Equation (10.1.4)](01-prologue.md#eq-general-cm-loss), the intractable oracle target ${\boldsymbol{\Psi}}_{s\to 0}$ is replaced by a teacher–induced surrogate that uses the *semigroup property* of the ODE flow (see more details later in [Equation (11.2.1)](../11-fast-generators/02-discrete-consistency-models.md#eq-semi-group)): evolving from $s$ to $t$ should be equivalent to going from $s$ to any intermediate $u$ and then from $u$ to $t$,

$$
{\boldsymbol{\Psi}}_{s\to t} = {\boldsymbol{\Psi}}_{u\to t}\circ{\boldsymbol{\Psi}}_{s\to u}.
$$

PD enforces this locally by training the student’s one–step map to match the teacher’s composition of two adjacent one–step fragments:

<span id="eq-pd-local"></span>

$$
\begin{aligned}
{\mathbb{E}}_{s}{\mathbb{E}}_{{\mathbf{x}}_s\sim p_s}
  \big\|
    \underbrace{{\mathbf{G}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s, s, s-2\Delta s)}_{\text{student one-step}}
    -
    \underbrace{\mathtt{Solver}_{s-\Delta s\to s-2\Delta s}\big(\mathtt{Solver}_{s\to s-\Delta s}({\mathbf{x}}_s)\big)}_{\text{teacher two-step composition}}
  \big\|_2^2.
\end{aligned}
\tag{10.3.3}
$$

When the teacher solver accurately approximates the underlying ODE flow, the composition in [Equation (10.3.3)](03-progressive-distillation.md#eq-pd-local) approximates the corresponding semigroup relation on a short decreasing grid (take $s>u>t$ with $u=s-\Delta s$ and $t=s-2\Delta s$):

$$
\begin{aligned}
{\boldsymbol{\Psi}}_{s\to s-2\Delta s}
  &= {\boldsymbol{\Psi}}_{s-\Delta s\to s-2\Delta s}\circ{\boldsymbol{\Psi}}_{s\to s-\Delta s}
  \\&\approx
  \mathtt{Solver}_{s-\Delta s\to s-2\Delta s}\circ \mathtt{Solver}_{s\to s-\Delta s},
\end{aligned}
$$

so training only requires short teacher fragments, rather than a full rollout from time $s$ all the way to $0$.

To connect back to the few–step denoiser view in [Equation (10.1.4)](01-prologue.md#eq-general-cm-loss), define the student’s few–step map as a composition of learned jumps:

$$
\underbrace{{\mathbf{G}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s, s, 0)}_{\text{few-step denoiser}}
=
\big({\mathbf{G}}_{{\boldsymbol{\theta}}}(\,\cdot\,, 2\Delta s, 0)\circ\cdots\circ {\mathbf{G}}_{{\boldsymbol{\theta}}}(\,\cdot\,, s, s-2\Delta s)\big)({\mathbf{x}}_s).
$$

Conceptually, [Equation (10.3.3)](03-progressive-distillation.md#eq-pd-local) provides an efficient *local surrogate* for the global regression

$$
{\mathbb{E}}_{s,{\mathbf{x}}_s}\Big\|{\mathbf{G}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s,0)\;-\;(\mathtt{Solver})^{\circ}_{s\to 0}({\mathbf{x}}_s)\Big\|_2^2,
$$

where $(\mathtt{Solver})^{\circ}_{s\to 0}$ denotes the teacher’s full composition from $s$ to $0$ on a grid with step size $\Delta s$, serving as a proxy for ${\boldsymbol{\Psi}}_{s\to 0}$.

**Can we Use Other Solvers?** Above, we used DDIM in the ${\mathbf{x}}$-prediction parameterization as a concrete PF–ODE sampler. The basic distillation principle is more general: several fine teacher steps produce a target endpoint, and the student is trained so that one coarser step reaches the same endpoint.

The closed-form pseudo-target derived above is available whenever the coarse solver update can be inverted with respect to the student’s network prediction. For DDIM, the update is affine in a single ${\mathbf{x}}$-prediction, so this inversion is analytic. The same idea applies to other solvers whose one-step update admits a comparable inversion.

For more complicated solvers, such a single-output inversion may not be available. Multi-stage methods use several network evaluations, multi-step methods depend on previous solver states, and implicit methods require solving coupled equations. In such cases, one can instead compute the endpoint of the coarse student step and directly match it to the endpoint produced by the fine teacher steps, as in [Equation (10.3.3)](03-progressive-distillation.md#eq-pd-local), backpropagating through the student solver when feasible. Since a coarse step may itself require multiple neural function evaluations, fewer time steps do not necessarily imply the same reduction in NFE.

For a stochastic sampler, the same pathwise logic applies after fixing or coupling the random draws used by the teacher and student. Without such coupling, individual trajectories need not share the same endpoint, and the appropriate target is generally distributional rather than pathwise matching.

## 10.3.4 PD with Guidance {#subsec-pd-guidance}

[Meng et al. (2023)](../../references.md#ref-meng2023distillation) proposed a two-stage pipeline for distilling classifier-free guided (CFG) diffusion models: (1) *distill the guidance* into a single network that takes the guidance weight as input, and (2) apply *progressive distillation (PD)* to reduce the sampling steps. They demonstrated this both in pixel space and in latent space (e.g., Stable Diffusion).

**Stage-One Distillation: Distilling Guidance.** Let ${\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_s, s, {\mathbf{c}})$ denote the (pre-trained) conditional diffusion model output in the “${\mathbf{x}}$-prediction” parameterization (i.e., a clean estimate) at time $s$ and condition ${\mathbf{c}}$; the condition can also be null, ${\mathbf{c}}=\emptyset$ (unconditional branch). The $\omega$-weighted CFG combination in [Equation (8.3.3)](../08-guidance/03-classifier-free-guidance.md#eq-cfg-model-nn) can be written as

<span id="eq-cfg-model-clean"></span>

$$
{\mathbf{x}}_{\boldsymbol{\phi}^\times}^{\,\omega}({\mathbf{x}}_s, s, {\mathbf{c}})
   := (1+\omega)\,{\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_s, s, {\mathbf{c}})
      - \omega\,{\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_s, s, \emptyset),
\tag{10.3.4}
$$

where $\omega \sim p_\omega(\omega)$ for some CFG weighting distribution $p_\omega$, typically $p_\omega(\omega)=\mathcal{U}[\omega_{\min},\omega_{\max}]$.

Stage-one introduces a new model ${\mathbf{x}}_{\boldsymbol{\theta}_1}({\mathbf{x}}_s, s, {\mathbf{c}}, \omega)$ that directly takes $\omega$ as input and learns to reproduce the CFG output ${\mathbf{x}}_{\boldsymbol{\phi}^\times}^{\,\omega}({\mathbf{x}}_s, s, {\mathbf{c}})$ by supervised regression:

$$
\min_{\boldsymbol{\theta}_1}\;
\mathbb{E}_{\omega\sim p_\omega, s, {\mathbf{x}}\sim p_{\mathrm{data}}, {\mathbf{x}}_s \sim p({\mathbf{x}}_s\mid {\mathbf{x}})}
\lambda(s)
\big\|{\mathbf{x}}_{\boldsymbol{\theta}_1}({\mathbf{x}}_s, s, {\mathbf{c}}, \omega)-{\mathbf{x}}_{\boldsymbol{\phi}^\times}^{\omega}({\mathbf{x}}_s, s, {\mathbf{c}})\big\|_2^2.
$$

Here $\lambda(s)$ is a standard schedule-dependent weighting; sampling $\omega$ each iteration teaches a single network to emulate CFG at arbitrary guidance strengths.

**Stage-Two Distillation: PD.** The stage-one model ${\mathbf{x}}_{\boldsymbol{\theta}_1}({\mathbf{x}}_s, s, {\mathbf{c}}, \omega)$ serves as the teacher in PD and is progressively distilled into a student ${\mathbf{x}}_{\boldsymbol{\theta}_2}({\mathbf{x}}_s, s, {\mathbf{c}}, \omega)$ with fewer sampling steps, following [Section 10.3.2](03-progressive-distillation.md#subsec-pd-training). At each iteration, the number of steps is halved (e.g., $N \to N/2 \to N/4 \to \cdots$).
