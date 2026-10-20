# 7.5 Is Diffusion Model's ODE an Optimal Map to OT Problem? {#sec-pfode-ot}

In this section, we focus on quadratic-cost optimal transport problem.

## 7.5.1 PF-ODE Flow Is Generally Not Optimal Transport {#subsec-pfode-not-optimal}

A natural question is whether the deterministic flow map induced by a diffusion process coincides with the optimal transport map under quadratic cost. The answer is no in general, and explicit counterexamples are already available in Gaussian settings. In particular, [Tanana (2017)](../../references.md#ref-tanana2017comparison) showed that, for a diffusion flow with a more general Gaussian reference, the resulting deterministic transport map can differ from the Brenier optimal transport map even when the endpoint distributions are Gaussian. Since both maps are explicit in this setting, their difference can be verified directly.

Below we present the stronger construction of [Lavenant and Santambrogio (2022)](../../references.md#ref-lavenant2022flow) in the standard-Gaussian Ornstein–Uhlenbeck setting that is directly relevant to diffusion models. Their argument establishes non-optimality for a broader class of initial distributions and provides structural insight into why the corresponding PF-ODE flow map need not be an optimal transport map.

<span id="fig-pfode-vs-ot-2d"></span>

![Probability flow ODE transport and quadratic-cost optimal transport can produce different maps](../../assets/chapter-07/pfode-vs-optimal-transport.png)

**Figure 7.6:** Comparing the PF-ODE flow map with the true OT map from $p_{\mathrm{data}}$ to $p_{\mathrm{prior}}$ in 2D. Right: the quadratic-cost OT map, computed exactly via the Hungarian algorithm. Left: the PF-ODE flow map obtained by integrating the PF-ODE. The two maps visibly differ, illustrating that the PF-ODE does not generally recover the optimal transport map.

Source: Created by the authors with AI-assisted coding.

**Setup.** We consider a VP SDE, specifically the Ornstein–Uhlenbeck process, which evolves a smooth initial density $p_0$ toward the standard Gaussian $\mathcal{N}(\boldsymbol{0}, {\mathbf{I}})$:

$$
\mathop{}\!\mathrm{d} {\mathbf{x}}(t) = -{\mathbf{x}}(t) \mathop{}\!\mathrm{d} t + \sqrt{2} \mathop{}\!\mathrm{d} {\mathbf{w}}(t), \quad {\mathbf{x}}(0) \sim p_0.
$$

The associated PF-ODE is given by

$$
\frac{\mathop{}\!\mathrm{d} \mathbf{S}_t({\mathbf{x}})}{\mathop{}\!\mathrm{d} t} = -\mathbf{S}_t({\mathbf{x}}) - \nabla \log p_t(\mathbf{S}_t({\mathbf{x}})), \quad \mathbf{S}_0({\mathbf{x}}) = {\mathbf{x}}.
$$

Here, $\mathbf{S}_t$ denotes the flow map pushing forward $p_0$ to the marginal $p_t$:

$$
\left(\mathbf{S}_t\right) \# p_0 = p_t,
\quad\text{that is,}\quad
p_t({\mathbf{y}}) = \int_{\mathbb{R}^D} \delta({\mathbf{y}} - \mathbf{S}_t({\mathbf{x}}))   p_0({\mathbf{x}}) \mathop{}\!\mathrm{d} {\mathbf{x}}.
$$

These densities $p_t$ evolve via the Fokker-Planck equation:

$$
\frac{\partial p_t}{\partial t} = \nabla \cdot ({\mathbf{x}} p_t) + \Delta p_t.
$$

This is equivalent to a continuity equation with velocity field:

$$
{\mathbf{v}}_t({\mathbf{x}}) = -{\mathbf{x}} - \nabla \log p_t({\mathbf{x}}),
$$

whose flow is given by $\mathbf{S}_t({\mathbf{x}})$. In other words, the PF-ODE can be written as:

$$
\frac{\mathop{}\!\mathrm{d} \mathbf{S}_t({\mathbf{x}})}{\mathop{}\!\mathrm{d} t} = {\mathbf{v}}_t\left(\mathbf{S}_t({\mathbf{x}})\right).
$$

As $t \to \infty$, the map transports the initial distribution to the prior:

$$
\mathbf{S}_\infty \# p_0 = \mathcal{N}(\boldsymbol{0}, {\mathbf{I}}) =: p_{\text{prior}}.
$$

**Objective of [Lavenant and Santambrogio (2022)](../../references.md#ref-lavenant2022flow)’s Argument.**

[Lavenant and Santambrogio (2022)](../../references.md#ref-lavenant2022flow) do not directly assess whether the terminal map $\mathbf{S}_\infty$ from $p_0$ to the Gaussian is optimal. Instead, they construct a specific initial distribution $p_0$ and examine the entire PF-ODE trajectory. Their key observation is that optimality may fail at some point along the flow.

They consider the intermediate marginal $p_t = \mathbf{S}_t \# p_0$ and define the residual transport map from $p_{t_0}$ to the Gaussian as

$$
\mathbf{T}_{t \to \infty} := \mathbf{S}_\infty \circ \mathbf{S}_{t}^{-1}, \quad \text{for all } t\geq 0.
$$

The core of their argument shows that, for a carefully chosen $p_0$, there exists a time $t_0 \geq 0$ such that $\mathbf{T}_{t_0 \to \infty}$ is not the quadratic-cost optimal transport map from the new starting distribution $p_{t_0}$ to $\mathcal{N}(\boldsymbol{0}, {\mathbf{I}})$.

This result demonstrates that PF-ODE flows do not, in general, yield optimal transport maps, and that the property of optimality can break down for certain initial distributions.

**Some Tools.** The argument of [Lavenant and Santambrogio (2022)](../../references.md#ref-lavenant2022flow) crucially relies on the following result, known as *Brenier’s theorem*:

<span id="thm-brenier"></span>

**Theorem 7.1: Informal Brenier’s Theorem**

Let $\nu_1, \nu_2$ be two probability distributions on $\mathbb{R}^D$ with smooth densities. A smooth map ${\mathbf{T}} : \mathbb{R}^D \to \mathbb{R}^D$ is the optimal transport from $\nu_1$ to $\nu_2$ (under quadratic cost) if and only if ${\mathbf{T}} = \nabla u$ for some convex function $u$. In this case, $\mathrm{D} {\mathbf{T}}$ is symmetric and positive semi-definite, and $u$ satisfies the Monge–Ampère equation:

$$
\det \mathrm{D}^2 u(\mathbf{x}) = \frac{\nu_1(\mathbf{x})}{\nu_2(\nabla u(\mathbf{x}))}.
$$

The proof also implicitly uses the following fact, which we will not repeat each time: *a map is the optimal transport between two distributions if and only if its inverse is the optimal transport in the reverse direction*.

**Proof Sketch: PF-ODE Is Not an OT Map in General.** [Lavenant and Santambrogio (2022)](../../references.md#ref-lavenant2022flow) employ a proof by contradiction: they assume that for every $t\ge0$, the map

$$
\mathbf{T}_t  =  \mathbf{S}_t \circ \mathbf{S}_\infty^{-1}
$$

is the quadratic-cost optimal transport map from $\mathcal{N}(\boldsymbol{0},{\mathbf{I}})$ to $p_t$.

**Step 1: Brenier’s Theorem.** By Brenier’s Theorem, the Jacobian of any optimal transport map from Gaussian must be symmetric and positive semi-definite. Thus,

$$
\mathrm{D} \mathbf{T}_t({\mathbf{x}}) = \mathrm{D} \mathbf{S}_t(\mathbf{S}_\infty^{-1}({\mathbf{x}})) \mathrm{D} (\mathbf{S}_\infty^{-1})({\mathbf{x}})
$$

must be symmetric for all $t$ and ${\mathbf{x}}$. Here, $\mathrm{D} \mathbf{T}_t({\mathbf{x}})$ denotes the total differentiation with respect to ${\mathbf{x}}$.

**Step 2: Time-Differentiating the Symmetry Condition.** Differentiating in time:

$$
\frac{\partial}{\partial t} \mathrm{D}\mathbf{T}_t({\mathbf{x}})
= \left( \frac{\partial}{\partial t} \mathrm{D} \mathbf{S}_t \right)(\mathbf{S}_\infty^{-1}({\mathbf{x}})) \mathrm{D} (\mathbf{S}_\infty^{-1})({\mathbf{x}}).
$$

Given that the symmetry holds for all $t$, it follows that this derivative remains symmetric.

Using the flow ODE (differentiating in ${\mathbf{x}}$), we obtain:

$$
\frac{\partial (\mathrm{D}\mathbf{S}_t)}{\partial t}
= \mathrm{D}{\mathbf{v}}_t(\mathbf{S}_t) \cdot \mathrm{D}\mathbf{S}_t
= \left( -{\mathbf{I}} - \mathrm{D}^2 \log p_t(\mathbf{S}_t) \right) \cdot \mathrm{D}\mathbf{S}_t.
$$

Combining the above, we see that

$$
\left( -{\mathbf{I}} - \mathrm{D}^2 \log p_t(\mathbf{S}_t) \right) \cdot\mathrm{D}\mathbf{S}_t \cdot \mathrm{D} (\mathbf{S}_\infty^{-1})
$$

is symmetric for all $t\geq 0$.

At $t = 0$, we have $\mathbf{S}_0 = {\mathbf{I}}$ and $\mathrm{D}\mathbf{S}_0 = {\mathbf{I}}$, yielding:

$$
\left( -{\mathbf{I}} - \mathrm{D}^2 \log p_0(\mathbf{S}_\infty^{-1}({\mathbf{x}})) \right) \cdot \mathrm{D} (\mathbf{S}_\infty^{-1})({\mathbf{x}}) \quad \text{is symmetric}.
$$

**Step 3: The Commutation Condition.** Since $\mathbf{T}_0 = \mathbf{S}_\infty^{-1}$ is assumed to be optimal, its Jacobian $D\mathbf{T}_0 = \mathrm{D}(\mathbf{S}_\infty^{-1})$ is symmetric. Moreover, the Hessian $\mathrm{D}^2\log p_0$ is symmetric. Recall that two symmetric matrices multiply to a symmetric matrix if and only if they commute. Hence, for all $\mathbf{x}\in\mathbb{R}^D$,

$$
\mathrm{D}^2\log p_0\bigl(\mathbf{S}_\infty^{-1}(\mathbf{x})\bigr)
\quad\text{must commute with}\quad
\mathrm{D}(\mathbf{S}_\infty^{-1})(\mathbf{x}) .
$$

Setting $\mathbf{y}=\mathbf{S}_\infty^{-1}(\mathbf{x})$ gives the equivalent condition: for all ${\mathbf{y}}\in\mathbb{R}^D$,

$$
\mathrm{D}^2\log p_0(\mathbf{y})
\quad\text{must commute with}\quad
\mathrm{D}\mathbf{S}_\infty(\mathbf{y}) .
$$

Now, we transform this condition into a more computable form. Since $\mathbf{S}_\infty$ is optimal between $p_0$ and $\mathcal{N}(\boldsymbol{0},{\mathbf{I}})$, Brenier’s theorem guarantees that $\mathbf{S}_\infty = \nabla u$ for some convex function $u$. From the Monge–Ampère equation, it follows that:

$$
\log p_0({\mathbf{y}}) = \log \det(\mathrm{D}^2 u({\mathbf{y}})) - \frac{1}{2} \|\nabla u({\mathbf{y}})\|^2 + \text{Constant}.
$$

The condition becomes (with $\mathrm{D}{\mathbf{S}}_\infty = \mathrm{D}^2 u$):

<span id="eq-necessary-ot"></span>

$$
\begin{aligned}
\mathrm{D}^2 \left(\log \det \mathrm{D}^2 u - \frac{1}{2} \|\nabla u\|^2\right)
\quad\text{must commute with}\quad
\mathrm{D}^2 u.
\end{aligned}
\tag{7.5.1}
$$

This yields a necessary condition for $\mathbf{T}_t$ to be optimal.

**Step 4: Constructing the Counterexample.** Let us show how to leverage this necessary condition to derive a contradiction.

Assume we can construct a convex function $u$ such that

$$
\mathrm{D}^2 \left( \log \det \mathrm{D}^2 u(\mathbf{x}) - \frac{1}{2} \lvert \nabla u(\mathbf{x}) \rvert^2 \right)
$$

does not commute with $\mathrm{D}^2 u(\mathbf{x})$ for some $\mathbf{x} \in \mathbb{R}^D$. Defining $p_0 = (\nabla u)^{-1} \# \mathcal{N}(\boldsymbol{0}, {\mathbf{I}})$, Brenier’s theorem implies that $\nabla u$ is the optimal transport from $p_0$ to $\mathcal{N}(\boldsymbol{0}, {\mathbf{I}})$. However, the condition in [Equation (7.5.1)](05-probability-flow-odes-and-optimal-transport.md#eq-necessary-ot) fails, leading to a contradiction. Thus, our goal is to construct such a function. Consider

$$
u(\mathbf{x}) = \frac{1}{2} \|\mathbf{x}\|^2 + \varepsilon \phi(\mathbf{x}), \quad \text{for a small } \varepsilon.
$$

Then $\mathrm{D}^2 u(\mathbf{0}) = {\mathbf{I}} + \varepsilon \mathrm{D}^2 \phi(\mathbf{0})$, and the commutation condition at $\mathbf{x} = \mathbf{0}$ requires $\mathrm{D}^2 \phi(\mathbf{0})$ to commute with $\mathrm{D}^2 (\Delta \phi)(\mathbf{0})$.

For example, in $\mathbb{R}^2$, choosing

$$
\phi(x_1, x_2) = x_1 x_2 + x_1^4
$$

provides a counterexample where the Hessian $\mathrm{D}^2 \log p_0$ and the Jacobian $\mathrm{D}^2 u$ do not commute.

This contradiction shows that $\mathbf{T}_t$ cannot be optimal for all $t \geq 0$. Therefore, there exists some $t_0 \geq 0$ such that the map $\mathbf{T}_{t_0 \to \infty}$ is not optimal.

## 7.5.2 Can Canonical Linear Flow and Reflow Leads to an OT Map? {#subsec-ot-linear-reflow}

We have seen that the PF-ODE (especially in VP type forward kernel) is generally not an OT map. One natural question now is:

<span id="question-7-5-1"></span>

**Question 7.5.1**

Does the linear interpolation flow $(1-t){\mathbf{x}}_0 + t {\mathbf{x}}_1$ with ${\mathbf{x}}_0\sim p_{\mathrm{src}}$ and ${\mathbf{x}}_1\sim p_{\mathrm{tgt}}$, when applied to the independent coupling $\pi({\mathbf{x}}_0, {\mathbf{x}}_1) = p_{\mathrm{src}}({\mathbf{x}}_0)p_{\mathrm{tgt}}({\mathbf{x}}_1)$, recover the OT map?

The answer to the question is no.

Nevertheless, combining a linear path with a given coupling offers a practical upper bound on the true OT cost. Among all possible paths, linear interpolation provides the tightest such upper bound, as we will see in the following discussion.

**Canonical Linear Flow and Optimal Transport.** Focusing on optimal transport with quadratic cost, we consider the equivalent form of [Equation (7.2.1)](02-transport-problem-setups.md#eq-ot), the Benamou–Brenier formulation in [Equation (7.2.3)](02-transport-problem-setups.md#eq-bb-ot):

$$
\mathcal{K}\left(p_{\mathrm{src}}, p_{\mathrm{tgt}}\right) :=  \min_{\substack{ (p_t, {\mathbf{v}}_t) \text{ s.t. }
        \partial_t p_t + \nabla \cdot (p_t {\mathbf{v}}_t) = 0, \\
        p_0 = p_{\mathrm{src}},\,\, p_1 = p_{\mathrm{tgt}}
    }}
    \int_0^1 \int_{\mathbb{R}^D} \|{\mathbf{v}}_t({\mathbf{x}})\|^2 p_t({\mathbf{x}}) \mathop{}\!\mathrm{d}{\mathbf{x}} \mathop{}\!\mathrm{d} t.
$$

Unlike the static Monge formulation ([Equation (7.2.2)](02-transport-problem-setups.md#eq-monge-formulation)), where computing the optimal map requires solving the Monge–Ampère equation ([Equation (7.2.4)](02-transport-problem-setups.md#eq-monge-ampere)), the Benamou–Brenier problem admits a convex reformulation after introducing the momentum field ${\mathbf{m}}_t := p_t {\mathbf{v}}_t$. Nevertheless, discretizing this convex problem on a grid in $\mathbb{R}^D$ remains intractable in high dimensions.

While solving the Benamou-Brenier formulation is generally intractable, [Liu (2022)](../../references.md#ref-liu2022rectified) and [Lipman et al. (2024)](../../references.md#ref-lipman2024flow) reveal that its kinetic energy admits a practical upper bound. This is achieved by restricting the search to a simpler family of *conditional flows*, where each path is defined by its fixed endpoints $({\mathbf{x}}_0, {\mathbf{x}}_1)$ drawn from a coupling $\pi_{0,1}$ of the source and target distributions. Within this *conditional flow* family, the canonical linear interpolation emerges as the optimal choice, as formalized below.

<span id="ot-conditional-flow"></span>

**Proposition 7.5.1: An Upper Bound on OT Kinetic Energy via Conditional Flows**

Let $\pi_{0,1}$ be any coupling between $p_{\mathrm{src}}$ and $p_{\mathrm{tgt}}$.

-   **(1)** The kinetic energy is bounded above by the expected path energy of any conditional flow $\boldsymbol{\Psi}_t({\mathbf{x}}_0, {\mathbf{x}}_1)$ that connects the endpoints:

    $$
    \mathcal{K}\left(p_{\mathrm{src}}, p_{\mathrm{tgt}}\right) \le \mathbb{E}_{({\mathbf{x}}_0, {\mathbf{x}}_1) \sim \pi_{0,1}} \left[ \int_0^1 \|\boldsymbol{\Psi}'_t({\mathbf{x}}_0, {\mathbf{x}}_1)\|^2 \mathop{}\!\mathrm{d} t \right].
    $$

-   **(2)** The unique conditional flow $\boldsymbol{\Psi}^*_t$ that minimizes the upper bound on the right-hand side is the linear interpolation path:

    $$
    \boldsymbol{\Psi}^*_t({\mathbf{x}}_0, {\mathbf{x}}_1) = (1 - t) {\mathbf{x}}_0 + t {\mathbf{x}}_1.
    $$

    Substituting this optimal path yields the tightest version of the bound:

    $$
    \mathcal{K}\left(p_{\mathrm{src}}, p_{\mathrm{tgt}}\right) \le \mathbb{E}_{({\mathbf{x}}_0, {\mathbf{x}}_1) \sim \pi_{0,1}} \|{\mathbf{x}}_1 - {\mathbf{x}}_0\|^2.
    $$

**Proof for Proposition.**

The proof relies on a straightforward application of Jensen’s inequality and the tower property of conditional expectations, before solving the simplified variational problem with the Euler-Lagrange equation; we refer to ([Lipman et al., 2024](../../references.md#ref-lipman2024flow))’s Section 4.7 for the complete argument.

In other words, the linear interpolation $\boldsymbol{\Psi}^*_t$ (i.e., the forward kernel used by Flow Matching and Rectified Flow) minimizes an upper bound on the true kinetic energy for any chosen coupling $\pi_{0,1}$.

We emphasize that optimality within this class of conditional flows does not guarantee global optimality on the marginal distributions.

**Reflow and Optimal Transport.** The most naive transport plan between two distributions is to connect their samples with straight lines using a simple independent coupling. However, this approach is demonstrably not optimal, as the failure lies not in the straight-line paths themselves, but in the inefficient initial pairing of points.

The Reflow procedure may offer a constructive response. It is an iterative algorithm designed specifically to correct this pairing, and crucially, each step is guaranteed to be cost-non-increasing ([Liu et al., 2022](../../references.md#ref-liu2022flow)). This property suggests Reflow systematically pushes the transport plan towards a more optimal configuration, which naturally motivates the central question of its convergence.

<span id="question-7-5-2"></span>

**Question 7.5.2**

What happens if we apply the `Rectify` operator iteratively? Can the resulting sequence of transport plans converge to the optimal one, or does the fixed point of the Reflow process yield the OT map?

The short answer is no in general. Below, we explain what may go wrong. To recall, the Reflow procedure iteratively refines the coupling between $p_{\mathrm{src}}$ and $p_{\mathrm{tgt}}$ via the update:

$$
\pi^{(k+1)} = \texttt{Rectify}(\pi^{(k)}),
$$

initialized with the product coupling $\pi^{(0)} := p_{\mathrm{src}}({\mathbf{x}}_0) p_{\mathrm{tgt}}({\mathbf{x}}_1)$. More precisely, $\texttt{Rectify}$ output the updated coupling $\pi^{(k+1)}$ via the following: At each iteration $k = 0, 1, 2, \ldots$, a velocity field ${\mathbf{v}}_t^{(k)}$ is learned via:

$$
{\mathbf{v}}_t^{(k)} \in \operatorname*{arg\,min}_{{\mathbf{u}}_t} \, \mathcal{L}({\mathbf{u}}_t\big\vert\pi^{(k)}),
$$

where $\mathcal{L}({\mathbf{u}}_t\big\vert\pi^{(k)})$ is the loss (e.g., RF or FM loss) defined in [Equation (5.4.1)](../05-flow-based-perspective/04-canonical-affine-flow.md#eq-reflow-loss). Here, for notational simplicity, we adopt a non-parametric formulation for the velocity field, rather than a parameterized form $\boldsymbol{\phi}$ employed in other contexts. The updated coupling is then given by:

$$
\pi^{(k+1)}({\mathbf{x}}_0, {\mathbf{x}}_1) := p_{\mathrm{src}}({\mathbf{x}}_0)\, \delta\big({\mathbf{x}}_1 - \boldsymbol{\Psi}_1^{(k)}({\mathbf{x}}_0)\big),
$$

where $\boldsymbol{\Psi}_1^{(k)}$ denotes the solution map at time $t=1$ obtained by integrating ${\mathbf{v}}_t^{(k)}$ from initial condition ${\mathbf{x}}_0$.

It has been observed in ([Liu et al., 2022](../../references.md#ref-liu2022flow)) that for a coupling $\pi$ between $p_{\mathrm{src}}$ and $p_{\mathrm{tgt}}$, the existence of a velocity field ${\mathbf{v}}_t$ that minimizes the Reflow loss, that is, satisfies $\mathcal{L}({\mathbf{v}}_t |\pi) = 0$, does not necessarily imply that the transport is optimal.

Motivated by the Benamou–Brenier framework, where the optimal transport velocity is known to be the gradient of a potential function, [Liu (2022)](../../references.md#ref-liu2022rectified) proposed an additional constraint: the velocity field ${\mathbf{v}}_t$ should be a potential field. Accordingly, the objective in [Equation (5.4.1)](../05-flow-based-perspective/04-canonical-affine-flow.md#eq-reflow-loss) is modified to restrict ${\mathbf{v}}_t$ to the space of gradient vector fields, also known as potential flows:

<span id="eq-reflow-loss-gradient"></span>

$$
\begin{aligned}
{\mathbf{w}}_t^{(k)} \in \operatorname*{arg\,min}_{\substack{{\mathbf{u}}_t:\, {\mathbf{u}}_t=\nabla\varphi \\ \text{for some }\varphi\colon\mathbb{R}^D\to\mathbb{R}}} \, \mathcal{L}({\mathbf{u}}_t\big\vert\pi^{(k)}),
\end{aligned}
\tag{7.5.2}
$$

with the rest of the procedure remaining the same as in `Rectify`. We denote this associated operator as $\texttt{Rectify}_{\perp}$, emphasizing the projection onto irrotational vector fields.

Let $\pi$ be a coupling between $p_{\mathrm{src}}$ and $p_{\mathrm{tgt}}$. [Liu et al. (2022)](../../references.md#ref-liu2022flow) conjecture the following equivalence characterizing optimality:

-   **(i)** $\pi$ *is an optimal transport coupling.*

-   **(ii)** $\pi$ *is a fixed point of the potential rectification operator:*

    $$
    \pi = \texttt{Rectify}_{\perp}(\pi).
    $$

-   **(iii)** *There exists a gradient velocity field ${\mathbf{v}}_t = \nabla \varphi_t$ such that the rectify loss vanishes:*

    $$
    \mathcal{L}({\mathbf{v}}_t |\pi) = 0.
    $$

However, [Hertrich et al. (2025)](../../references.md#ref-hertrich2025relation) exhibit two types of counterexamples:

1.  When the intermediate distributions $p_t$ have disconnected support, one can find fixed points of $\texttt{Rectify}_{\perp}$ with zero Reflow loss and gradient velocity fields that nonetheless fail to produce the optimal coupling.

2.  Even when both endpoint distributions are Gaussian, there exist couplings whose loss is arbitrarily small but whose deviation from the optimal coupling is arbitrarily large.

Therefore, while rectified flows may yield strong generative models, their reliability as optimal transport solvers remains limited. This highlights an important gap between generative modeling and principled optimal transport theory, inviting further research at their intersection.

Finally, we note that transport cost does not always correlate with downstream performance; as such, computing the exact optimal transport map may not necessarily lead to better practical outcomes. Nonetheless, variants of optimal transport remain fundamental to many problems in science and engineering. Diffusion models offer a powerful framework for exploring these challenges.
