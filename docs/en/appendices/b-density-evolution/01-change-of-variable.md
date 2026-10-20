# B.1 Change-of-Variable Formula: From Deterministic Maps to Stochastic Flows {#sec-cov}

In this section, we aim to demystify the continuity equation and the Fokker-Planck equation by drawing analogies to the classic change-of-variable formula from calculus. We begin with the familiar single-variable case, extend it to the multivariate setting and to probability densities ([Section B.1.1](01-change-of-variable.md#subsec-cov-det)), then generalize to compositions of bijective maps whose continuous-time limit leads to the continuity equation ([Section B.1.2](01-change-of-variable.md#subsec-cov-continuity-eq)). Finally, we incorporate stochasticity by introducing random noise, which naturally extends the continuity equation to the Fokker-Planck equation ([Section B.1.3](01-change-of-variable.md#subsec-cov-fpe)).

## B.1.1 Change-of-Variable Formula for Deterministic Maps {#subsec-cov-det}

We move particles according to a deterministic map and study how their *law* (density) evolves. The key principle is conservation of probability mass, grounded in a fundamental result from calculus and probability: the change-of-variable formula. This formula describes how integrals, and therefore probability densities, transform under smooth bijective mappings. To build intuition, we first consider a single update step, and then extend the discussion to sequential transformations.

**Single Update.** Think of a single update rule induced by applying a vector field (analogous to a force) $\boldsymbol{\Psi}:{\mathbb{R}}^D \to {\mathbb{R}}^D$ for one unit of time. Starting from an initial particle state ${\mathbf{x}}_0$, its next state is given by

$$
{\mathbf{x}}_1 = \boldsymbol{\Psi}({\mathbf{x}}_0).
$$

**Underlying Pattern (a Density) and How it Moves.** If the initial states follow an underlying “law/pattern” described by a density $p_0$ (i.e., ${\mathbf{x}}_0\sim p_0$), then applying $\boldsymbol{\Psi}$ produces a new density $p_1$ for ${\mathbf{x}}_1$ (i.e., ${\mathbf{x}}_1\sim p_1$). Assuming $\boldsymbol{\Psi}$ is a smooth bijection, $p_1$ is obtained from $p_0$ via the standard *change-of-variables formula*:

<span id="eq-change-of-variable-density"></span>

$$
p_{1}({\mathbf{x}}_1) = p_{0}(\boldsymbol{\Psi}^{-1}({\mathbf{x}}_1)) \cdot \left| \det \left( \frac{\partial \boldsymbol{\Psi}^{-1}}{\partial {\mathbf{x}}_1} \right) \right|.
\tag{B.1.1}
$$

Here $\tfrac{\partial \boldsymbol{\Psi}}{\partial {\mathbf{x}}}$ is the Jacobian matrix of $\boldsymbol{\Psi}$, denoted $\partial_{\mathbf{x}} \boldsymbol{\Psi}$. Equivalently, in the original coordinates,

$$
p_0({\mathbf{x}}_0) \;=\; p_1\!\big(\boldsymbol{\Psi}({\mathbf{x}}_0)\big)\;\big|\det \partial_{\mathbf{x}} \boldsymbol{\Psi}({\mathbf{x}}_0)\big|.
$$

In words, $\boldsymbol{\Psi}$ reshapes the density $p_0$ into $p_1$. The factor $\big|\det \partial_{\mathbf{x}} \boldsymbol{\Psi}\big|$ represents the local change in volume; since probability mass is conserved, the density compensates by its inverse.

As a simple case, if $\boldsymbol{\Psi}$ is linear with an invertible matrix ${\mathbf{A}}$ (i.e., ${\mathbf{x}}_1={\mathbf{A}}{\mathbf{x}}_0$), then

$$
p_1({\mathbf{x}}_1)\;=\;p_0({\mathbf{A}}^{-1}{\mathbf{x}}_1)\,\big|\det {\mathbf{A}}^{-1}\big|.
$$

Schematically, we can read it as:

$$
\begin{array}{lccc}
   \textbf{Sample:} & {\mathbf{x}}_0 & \xrightarrow{\hspace{0.2cm} \boldsymbol{\Psi} \hspace{0.2cm}} & {\mathbf{x}}_1  \\
   \textbf{Density:} & p_{{\mathbf{x}}_0}({\mathbf{x}}_0) & \xrightarrow{\hspace{0.2cm} \boldsymbol{\Psi} \hspace{0.2cm}} & p_{{\mathbf{x}}_1}({\mathbf{x}}_1)
\end{array}
$$

**Why is [Equation (B.1.1)](01-change-of-variable.md#eq-change-of-variable-density) the Change-of-Variables Formula?** This comes directly from the familiar rule in calculus.

**Single-Variable Case.** Let $y = \Psi(x)$ be smooth and invertible. Rewriting an integral over $y$ in terms of $x$ gives

$$
\int g(y)\,\mathop{}\!\mathrm{d} y = \int g(\Psi(x)) \cdot |\Psi'(x)|\,\mathop{}\!\mathrm{d} x,
$$

where $|\Psi'(x)|$ compensates for interval stretching or compression, ensuring area preservation.

**Multivariate Case.** For $\boldsymbol{\Psi}: {\mathbb{R}}^D \to {\mathbb{R}}^D$ with ${\mathbf{y}}=\boldsymbol{\Psi}({\mathbf{x}})$,

$$
\int g({\mathbf{y}})\,\mathop{}\!\mathrm{d}{\mathbf{y}}
= \int g(\boldsymbol{\Psi}({\mathbf{x}})) \,\big|\det(\partial_{\mathbf{x}} \boldsymbol{\Psi})\big| \,\mathop{}\!\mathrm{d}{\mathbf{x}},
$$

so infinitesimal volumes transform as

$$
\mathop{}\!\mathrm{d} {\mathbf{y}} = \big|\det(\partial_{\mathbf{x}} \boldsymbol{\Psi})\big|\,\mathop{}\!\mathrm{d} {\mathbf{x}}.
$$

From this, the density formula in [Equation (B.1.1)](01-change-of-variable.md#eq-change-of-variable-density) follows:

$$
\begin{aligned}
p_{{\mathbf{y}}}({\mathbf{y}})
= \int_{{\mathbb{R}}^D} \delta({\mathbf{y}} - \boldsymbol{\Psi}({\mathbf{x}}))\, p_{{\mathbf{x}}}({\mathbf{x}})\mathop{}\!\mathrm{d} {\mathbf{x}} = p_{{\mathbf{x}}}\!\big(\boldsymbol{\Psi}^{-1}({\mathbf{y}})\big)\,\Big|\det\!\left(\frac{\partial \boldsymbol{\Psi}^{-1}}{\partial {\mathbf{y}}}\right)\Big|.
\end{aligned}
$$

**Composing Multiple Bijections.** We now apply several updates in sequence. Let ${\mathbf{x}}_k = \boldsymbol{\Psi}_k({\mathbf{x}}_{k-1})$ for $k=1,\dots,L$; that is,

$$
{\mathbf{x}}_0 \xrightarrow{\ \boldsymbol{\Psi}_1\ } {\mathbf{x}}_1 \xrightarrow{\ \boldsymbol{\Psi}_2\ } \cdots
\xrightarrow{\ \boldsymbol{\Psi}_L\ } {\mathbf{x}}_L,
$$

where each $\boldsymbol{\Psi}_k:{\mathbb{R}}^D \to {\mathbb{R}}^D$ is a smooth bijection. If the initial state follows density $p_0$ (i.e., ${\mathbf{x}}_0 \sim p_0$), then the sequence of updates induces densities $p_1,\dots,p_L$ for ${\mathbf{x}}_1,\dots,{\mathbf{x}}_L$.

Because probability mass is conserved at each step, the densities evolve according to

$$
p_{k}({\mathbf{x}}_k) \;=\; p_{k-1}({\mathbf{x}}_{k-1}) \,
   \Big|\det \partial_{{\mathbf{x}}_{k-1}} \boldsymbol{\Psi}_k({\mathbf{x}}_{k-1})\Big|^{-1},
   \quad k=1,\dots,L.
$$

By recursion, the final density at ${\mathbf{x}}_L$ is

<span id="eq-cov-multi-layer-nolog"></span>

$$
\begin{aligned}
p_{{\mathbf{x}}_L}({\mathbf{x}}_L)
    = p_{{\mathbf{x}}_0}({\mathbf{x}}_0)
    \cdot \prod_{k=1}^L \left| \det \left( \frac{\partial \boldsymbol{\Psi}_k}{\partial {\mathbf{x}}_{k-1}} \right) \right|^{-1}.
\end{aligned}
\tag{B.1.2}
$$

Equivalently, in log-density form:

$$
\begin{aligned}
\log p_{{\mathbf{x}}_L}({\mathbf{x}}_L)
    = \log p_{{\mathbf{x}}_0}({\mathbf{x}}_0)
    - \sum_{k=1}^L \log \left| \det \left( \frac{\partial \boldsymbol{\Psi}_k}{\partial {\mathbf{x}}_{k-1}} \right) \right|.
\end{aligned}
$$

This expression reflects how each transformation $\boldsymbol{\Psi}_k$ stretches or contracts volume, as captured by the Jacobian determinant. The accumulation of these local volume changes along the transformation path determines the final probability density under the composed map.

[Equation (B.1.2)](01-change-of-variable.md#eq-cov-multi-layer-nolog) serves as the core principle underlying Normalizing Flows (see [Section 5.1.2](../../chapters/05-flow-based-perspective/01-normalizing-flows-and-neural-odes.md#subsec-node)).

## B.1.2 Continuous-Time Limit: Continuity Equation {#subsec-cov-continuity-eq}

We now pass from discrete updates to a continuous description. Suppose the particle motion is driven by a time-varying velocity field ${\mathbf{f}}:{\mathbb{R}}^D \times [0,T]\to {\mathbb{R}}^D$. Imagine evolving a particle ${\mathbf{x}}_0 \sim p_0$ through infinitely many small bijective updates. At each step $t$ of length $\Delta t>0$, the update is

$$
{\mathbf{x}}_{t+\Delta t} = \boldsymbol{\Psi}({\mathbf{x}}_t)
:= {\mathbf{x}}_t + \Delta t\, {\mathbf{f}}({\mathbf{x}}_t, t).
$$

As $\Delta t \to 0$, the composition of these updates converges to a continuous flow governed by a velocity field ${\mathbf{f}}:{\mathbb{R}}^D \times [0,T]\to {\mathbb{R}}^D$:

<span id="eq-ode-f-continuity"></span>

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t} = {\mathbf{f}}({\mathbf{x}}(t), t),
    \qquad {\mathbf{x}}(0) = {\mathbf{x}}_0 \sim p_0.
\end{aligned}
\tag{B.1.3}
$$

Under suitable smoothness assumptions (see [Appendix A](../a-differential-equations/index.md#app-de)), this ODE admits a unique solution for each initial condition, which defines a deterministic flow map $\boldsymbol{\Psi}_{0\to t}:{\mathbb{R}}^D \to {\mathbb{R}}^D$. In other words, $\boldsymbol{\Psi}_{0\to t}$ brings the initial state ${\mathbf{x}}_0$ to the solution of [Equation (B.1.3)](01-change-of-variable.md#eq-ode-f-continuity) at time $t$:

$$
\boldsymbol{\Psi}_{0\to t}({\mathbf{x}}_0)
= {\mathbf{x}}_0 + \int_{0}^{t} {\mathbf{f}}({\mathbf{x}}(\tau), \tau)\,\mathop{}\!\mathrm{d}\tau.
$$

As a result, the whole distribution also moves: the initial density $p_0$ is transported into the new density $p_t$, the law of ${\mathbf{x}}(t)$. Formally, this is written as a *pushforward*:

$$
p_t = \big(\boldsymbol{\Psi}_{0\to t}\big)_{\#} p_0.
$$

When $\boldsymbol{\Psi}_{0\to t}$ is smooth and invertible, this reduces to the familiar change-of-variables rule:

$$
p_t({\mathbf{x}})
= p_0\!\big(\boldsymbol{\Psi}_{t\to 0}({\mathbf{x}})\big)\,
   \big|\det \partial_{\mathbf{x}}\boldsymbol{\Psi}_{t\to 0}({\mathbf{x}})\big|
= \int \delta\!\left({\mathbf{x}} - \boldsymbol{\Psi}_{0\to t}({\mathbf{x}}_0)\right)
   p_0({\mathbf{x}}_0)\,\mathop{}\!\mathrm{d}{\mathbf{x}}_0.
$$

**Continuity Equation: How the Density Moves in Time.** Rather than writing a separate formula for the density at each time, we can describe how it moves continuously using a differential equation in space ${\mathbf{x}}$ and time $t$. The idea is simple: probability mass is conserved, and the velocity field ${\mathbf{f}}$ only redistributes it in space. This gives the *continuity equation*:

<span id="eq-continuity-cov"></span>

$$
\begin{aligned}
\frac{\partial}{\partial t} p_t({\mathbf{x}})
    + \nabla \cdot \big( p_t({\mathbf{x}})\,{\mathbf{f}}({\mathbf{x}}, t) \big) = 0.
\end{aligned}
\tag{B.1.4}
$$

Here the divergence term $\nabla \cdot (p_t {\mathbf{f}})$ measures how the flow locally expands or compresses the density, ensuring total probability remains $1$.

This partial differential equation (PDE) ensures that probability mass is conserved as the flow moves particles. In fact, it can be viewed as the continuous-time analogue of the change-of-variables formula.

**Derivation of Continuity Equation via Change-of-Variables Formula.** Conceptually, the continuity equation can also be obtained by taking the continuous-time limit of [Equation (B.1.2)](01-change-of-variable.md#eq-cov-multi-layer-nolog). Here, however, we adopt a more direct derivation based on [Equation (B.1.1)](01-change-of-variable.md#eq-change-of-variable-density).

**Discretization and Change-of-Variable Formula.** Consider

$$
\mathbf{x}_{t+\Delta t} := \boldsymbol{\Psi}(\mathbf{x}_t) = \mathbf{x}_t + \Delta t\, {\mathbf{f}}(\mathbf{x}_t, t),
$$

which is actually the forward Euler discretization of the ODE in [Equation (B.1.3)](01-change-of-variable.md#eq-ode-f-continuity) over a small time interval $\Delta t > 0$. The Jacobian of the map $\boldsymbol{\Psi}$ with respect to $\mathbf{x}_t$ expands as

$$
\frac{\partial \boldsymbol{\Psi}}{\partial \mathbf{x}_t} = {\mathbf{I}} + \Delta t \nabla_{\mathbf{x}} {\mathbf{f}}(\mathbf{x}_t, t) + \mathcal{O}(\Delta t^2),
$$

so its determinant satisfies

$$
\det\left( \frac{\partial \boldsymbol{\Psi}}{\partial \mathbf{x}_t} \right) = 1 + \Delta t\, \nabla \cdot {\mathbf{f}}(\mathbf{x}_t, t) + \mathcal{O}(\Delta t^2).
$$

This uses the standard expansion $\det({\mathbf{I}} + \Delta t\, \mathbf{A}) = 1 + \Delta t\, \operatorname{Tr}(\mathbf{A}) + \mathcal{O}(\Delta t^2)$ as $\Delta t \to 0$, along with $\nabla \cdot {\mathbf{f}} = \operatorname{Tr}(\nabla_{\mathbf{x}} {\mathbf{f}})$.

Applying the change-of-variables formula, the log-density evolves as

$$
\begin{aligned}
\log p_{t+\Delta t}(\mathbf{x}_{t+\Delta t}) = \log p_t(\mathbf{x}_t) - \Delta t\, \nabla \cdot {\mathbf{f}}(\mathbf{x}_t, t) + \mathcal{O}(\Delta t^2).
\end{aligned}
$$

That is,

<span id="eq-log-density-diff"></span>

$$
\begin{aligned}
\log p_{t+\Delta t}(\mathbf{x}_{t+\Delta t}) - \log p_t(\mathbf{x}_t) = - \Delta t\, \nabla \cdot {\mathbf{f}}(\mathbf{x}_t, t) + \mathcal{O}(\Delta t^2).
\end{aligned}
\tag{B.1.5}
$$

**Using Taylor Expansion.** Now, we expand the left-hand side via multivariate Taylor expansion:

$$
\begin{aligned}
&\log p_{t+\Delta t}({\mathbf{x}}_{t+\Delta t})
    - \log p_t({\mathbf{x}}_t)
    \\= &\Delta t\, \partial_t \log p_t({\mathbf{x}}_t)
    + ({\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t)^\top \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t)
    + \mathcal{O}(\Delta t^2).
\end{aligned}
$$

Substituting ${\mathbf{x}}_{t+\Delta t} - {\mathbf{x}}_t = {\mathbf{f}}({\mathbf{x}}_t, t)\Delta t$ yields:

$$
\begin{aligned}
&\log p_{t+\Delta t}({\mathbf{x}}_{t+\Delta t})
    - \log p_t({\mathbf{x}}_t)
    \\= & \Delta t\, \partial_t \log p_t({\mathbf{x}}_t)
    + \Delta t\, {\mathbf{f}}({\mathbf{x}}_t, t)^\top \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t)
    + \mathcal{O}(\Delta t^2).
\end{aligned}
$$

Matching terms with [Equation (B.1.5)](01-change-of-variable.md#eq-log-density-diff) and letting $\Delta t \to 0$, we conclude that

$$
\partial_t \log p_t({\mathbf{x}}_t)
    = - \nabla_{{\mathbf{x}}_t} \cdot {\mathbf{f}}({\mathbf{x}}_t, t)
    - {\mathbf{f}}({\mathbf{x}}_t, t)^\top \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t).
$$

Exponentiating and using the product rule yields the continuity equation.

**Lagrangian and Eulerian Views of Deterministic Transport.** The same deterministic transport admits two complementary descriptions. The *Lagrangian* view follows individual particles through the flow map $\boldsymbol{\Psi}_{0\to t}$, whereas the *Eulerian* view describes the velocity ${\mathbf{f}}({\mathbf{x}},t)$ and density $p_t({\mathbf{x}})$ at fixed spatial locations. Under suitable regularity, these descriptions are consistent: a prescribed velocity field generates a particle flow, whose pushforward density satisfies the continuity equation.

There is nevertheless an important asymmetry. Once the velocity field and initial density are fixed, the resulting flow and density evolution are determined. A prescribed density path alone, however, generally does not determine a unique velocity field or particle flow.

**From Velocity to Flow and Density.** So far, we have assumed that the velocity field ${\mathbf{f}}$ is given. The particle ODE

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t} = {\mathbf{f}}({\mathbf{x}}(t), t)
$$

describes how each particle moves, while the density PDE

$$
\partial_t p_t + \nabla \!\cdot \!\big(p_t {\mathbf{f}}_t\big) = 0
$$

describes how the entire distribution of particles evolves. These two views are connected: moving particles according to the ODE automatically produces a density that satisfies the PDE. In this case, the particle flow $\boldsymbol{\Psi}_{0\to t}$ is uniquely determined: starting from ${\mathbf{x}}(0)\sim p_0$, each trajectory ${\mathbf{x}}(t)$ is fixed, and the resulting density $p_t$ follows the continuity equation. Here, particle dynamics and density dynamics are fully consistent.

**From Density Path to Velocity: Non-Uniqueness.** If instead we begin only with the density path $t \mapsto p_t$ (e.g., [Section 5.3.2](../../chapters/05-flow-based-perspective/03-probability-paths-and-velocities.md#subsec-fm-instantiations) in flow matching), the velocity field is no longer uniquely determined. For example, if a vector field $\mathbf w_t$ satisfies

$$
\nabla_{{\mathbf{x}}}\cdot\!\big(p_t({\mathbf{x}})\,\mathbf w_t({\mathbf{x}})\big)=0
\qquad\text{(no net flux w.r.t. $p_t$),}
$$

then both ${\mathbf{f}}_t$ and ${\mathbf{f}}_t+\mathbf w_t$ give rise to the same density evolution. Thus a single density path may correspond to many different flows, and choosing one particular particle flow $\boldsymbol{\Psi}_{0\to t}$ amounts to picking a specific velocity field among these possibilities.

Not every given path $p_t$ can actually arise from particles moving under some velocity field. The continuity equation ([Equation (B.1.4)](01-change-of-variable.md#eq-continuity-cov)) provides the consistency check for whether a density path can be “generated by a flow”. We say that $p_t$ is *realizable* (or *generated by* ${\mathbf{f}}$) if there exists a velocity field ${\mathbf{f}}$ such that particles following

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t} = {\mathbf{f}}({\mathbf{x}}(t), t)
$$

produce exactly the densities $p_t$ through the flow map $\boldsymbol{\Psi}_{0\to t}$. That is, realizability holds when $p_t$ and ${\mathbf{f}}$ together satisfy [Equation (B.1.4)](01-change-of-variable.md#eq-continuity-cov).

Intuitively, realizability means that the snapshots of $p_t$ over time can be explained by particles moving under some velocity field, rather than being an arbitrary sequence of distributions.

When this condition holds, the density $p_t$ is nothing more than the pushforward of the initial density $p_0$ along the flow map $\boldsymbol{\Psi}_{0\to t}$. In this case, the familiar change-of-variables formula applies:

$$
\begin{aligned}
p_t = \big(\boldsymbol{\Psi}_{0\to t}\big)_{\#} p_0
= p_0\!\big(\boldsymbol{\Psi}_{t\to 0}({\mathbf{x}})\big)\,
   \big|\det \partial_{\mathbf{x}}\boldsymbol{\Psi}_{t\to 0}({\mathbf{x}})\big|
= \int \delta \left({\mathbf{x}} - \boldsymbol{\Psi}_{0\to t}({\mathbf{x}}_0)\right)
   p_0({\mathbf{x}}_0)\,\mathop{}\!\mathrm{d}{\mathbf{x}}_0.
\end{aligned}
$$

**(Optional) Conditioning.** If an additional conditioning variable ${\mathbf{z}} \sim \pi({\mathbf{z}})$ is introduced, the same reasoning applies for each fixed ${\mathbf{z}}$:

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t}={\mathbf{v}}_t({\mathbf{x}}(t)|{\mathbf{z}})
$$

with pushforward $p_t(\cdot|{\mathbf{z}})=({\boldsymbol{\Psi}}_{0\to t}(\cdot;{\mathbf{z}}))_{\#}p_0$, and continuity equation

$$
\partial_t p_t({\mathbf{x}}|{\mathbf{z}})+\nabla\!\cdot\!\big(p_t({\mathbf{x}}|{\mathbf{z}})\,{\mathbf{v}}_t({\mathbf{x}}|{\mathbf{z}})\big)=0
$$

The marginal density is then

$$
p_t({\mathbf{x}})=\int p_t({\mathbf{x}}|{\mathbf{z}})\pi({\mathbf{z}})\mathop{}\!\mathrm{d}{\mathbf{z}}.
$$

## B.1.3 Stochastic Processes: Fokker-Planck Equation {#subsec-cov-fpe}

When noise is added, the dynamics follow the SDE as in [Equation (A.2.3)](../a-differential-equations/02-stochastic-differential-equations.md#eq-sde):

$$
\mathop{}\!\mathrm{d} {\mathbf{x}}(t) = {\mathbf{f}}({\mathbf{x}}(t), t)\mathop{}\!\mathrm{d} t + g(t)\mathop{}\!\mathrm{d} {\mathbf{w}}(t).
$$

Then, the density $p_t({\mathbf{x}})$ satisfies the Fokker-Planck equation:

$$
\begin{aligned}
\frac{\partial p_t({\mathbf{x}})}{\partial t}
&= - \nabla \cdot \left( {\mathbf{f}}({\mathbf{x}}, t)\, p_t({\mathbf{x}}) \right)
+ \frac{1}{2} g^2(t)\, \Delta p_t({\mathbf{x}})
\\& = - \nabla \cdot \left(  \left({{\mathbf{f}}({\mathbf{x}}, t)}-{\frac{1}{2} g^2(t) \nabla_{\mathbf{x}} \log p_t({\mathbf{x}})}\right) p_t({\mathbf{x}})\right).
\end{aligned}
$$

$\Delta p_t = \nabla \cdot \nabla_{\mathbf{x}} p_t$ is the Laplacian operator. In the first equality, the first term describes transport of probability mass by the deterministic drift ${\mathbf{f}}$, while the second term models the spreading (diffusion) of the density due to stochastic noise with variance proportional to $\tfrac{1}{2} g^2(t)$. The second equality rewrites the same density evolution as a continuity equation with velocity

$$
{\mathbf{f}}({\mathbf{x}},t)
-
\frac{1}{2}g^2(t)\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}}).
$$

Therefore, the deterministic ODE driven by this velocity therefore evolves through the same one-time marginal densities $p_t$ as the SDE. This is the probability-flow ODE used throughout the book.

**Remark.**

The key point is that the SDE and PF-ODE agree on the *density path*, but generally not on how samples at different times are coupled. At any fixed time $t$, both produce the same distribution $p_t$; equivalently, their particle “snapshots” have the same histogram.

Across time, however, their sample paths are different. For the PF-ODE, once the state ${\mathbf{x}}_s$ is fixed, the later state is determined by ${\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)$. For the SDE, fresh Brownian noise is injected after time $s$, so the later state remains random even after conditioning on ${\mathbf{x}}_s$.

Consequently, the two processes may have different transition distributions, temporal correlations, and pathwise events, even though their marginal density at every individual time is identical. Thus, the PF-ODE reproduces the SDE’s *marginal density evolution*, rather than its full stochastic path law. See [Figure 4.5](../../chapters/04-score-sde-framework/02-matching-marginals.md#fig-score-sde-three-dynamics) for illustration.

The derivation of the Fokker-Planck equation is more involved; we refer the reader to [Section C.1.4](../c-ito-calculus/01-ito-formula.md#subsec-rigorous-proof-fpe).
