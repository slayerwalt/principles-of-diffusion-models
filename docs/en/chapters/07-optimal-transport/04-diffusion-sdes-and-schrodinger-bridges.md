# 7.4 Is Diffusion Model's SDE Optimal Solution to SB Problem? {#sec-dm-and-sb}

## 7.4.1 Diffusion models as a Special Case of Schrödinger Bridges {#subsec-diffusion-special-case-sb}

SB framework extends (score-based) diffusion models by enabling nonlinear interpolation between arbitrary source and target distributions. It achieves this by adding control drift terms derived from scalar potentials $\Psi(\mathbf{x}, t)$ and $\widehat{\Psi}(\mathbf{x}, t)$, which guide a reference diffusion process to match prescribed endpoint marginals (see [Equation (7.2.12)](02-transport-problem-setups.md#eq-pde-schrodinger)) and follow the decomposition:

$$
\nabla\log{\Psi(x,t)}+\nabla\log{\hat{\Psi}(x,t)}=\nabla \log p_t^{\mathrm{SB}}(\mathbf{x}).
$$

This generalization allows the model to move beyond standard Gaussian priors and generate samples from broader distributions.

**Connection to Diffusion Models.**

Diffusion models arise as a special case of the SB framework. Suppose the potential is constant, $\Psi(\mathbf{x}, t) \equiv 1$. Under this assumption, the second PDE in [Equation (7.2.12)](02-transport-problem-setups.md#eq-pde-schrodinger) reduces to the standard Fokker–Planck equation, whose solution is the marginal density of the reference process:

<span id="eq-sb-backward-potential"></span>

$$
\begin{aligned}
\widehat{\Psi}(\mathbf{x}, t) = p_t^{\mathrm{SB}}(\mathbf{x}).
\end{aligned}
\tag{7.4.1}
$$

The corresponding SB forward SDE thus becomes the uncontrolled reference process:

$$
\mathop{}\!\mathrm{d}\mathbf{x}_t = \mathbf{f}(\mathbf{x}_t, t)\mathop{}\!\mathrm{d} t + g(t)\mathop{}\!\mathrm{d}\mathbf{w}_t,
$$

and the SB backward SDE simplifies to:

$$
\mathop{}\!\mathrm{d}\mathbf{x}_t = \left[\mathbf{f}(\mathbf{x}_t, t) - g^2(t) \nabla \log p_t^{\mathrm{SB}}(\mathbf{x}_t) \right]\mathop{}\!\mathrm{d} t + g(t)\mathop{}\!\mathrm{d}\bar{\mathbf{w}}_t,
$$

which matches Anderson’s reverse-time SDE used in diffusion models. This correspondence shows that diffusion models can be interpreted as the zero-control limit of SB, where no additional drift is introduced by the potentials.

**Boundary Conditions and Generality.** The above reduction is purely formal unless the boundary constraints are compatible. For arbitrary source/target $(p_{\mathrm{src}},p_{\mathrm{tgt}})$, the PDE boundary conditions in [Equation (7.2.12)](02-transport-problem-setups.md#eq-pde-schrodinger) are generally not satisfied by the choice $\Psi\equiv 1$. Full SB resolves this by learning nontrivial potentials that induce a nonlinear control drift, bending the reference dynamics to match any prescribed endpoints. By contrast, diffusion models fix one endpoint to a simple prior (typically Gaussian) and learn only the reverse-time score to reach the data. With this perspective, SB is the more flexible umbrella: with nontrivial potentials it bridges arbitrary endpoints; with $\Psi\equiv 1$ it collapses to the diffusion-model case above. We additionally remark that in the standard linear diffusion model, $p_T \approx p_{\mathrm{prior}}$ holds only as $T \to \infty$, so the match to the prior is merely approximate.

## 7.4.2 Diffusion Models as Schrödinger Half-Bridges {#subsec-schrodinger-half-bridges}

In this section, we explain why diffusion models are not full Schrödinger bridges, but can instead be understood through the relaxed notion of *Schrödinger half-bridges*. A half-bridge enforces only one endpoint constraint (either $p_{\mathrm{prior}}$ or $p_{\mathrm{data}}$) rather than both, making it a one-sided variant of the full bridge. Before formalizing this connection, we introduce the definition of Schrödinger half-bridges, building on the general formulation in [Equation (7.2.7)](02-transport-problem-setups.md#eq-sb-epsilon-general) with arbitrary $p_{\mathrm{src}}$ and $p_{\mathrm{tgt}}$. We will then return to diffusion models and show how the half-bridge viewpoint naturally applies when the endpoints are given by $p_{\mathrm{prior}}$ and $p_{\mathrm{data}}$.

**Schrödinger Half-Bridges**

The SB problem asks for a stochastic process whose law is closest (in KL divergence) to a simple reference process, while *matching two endpoint distributions* $p_{\mathrm{src}}$ and $p_{\mathrm{tgt}}$. Solving the full bridge requires enforcing both boundary conditions, which is often computationally difficult. A useful relaxation is the *half-bridge* problem: instead of matching both endpoints, we match only one of them.

Formally, let $R$ be the reference path distribution. The *forward half-bridge* seeks a path distribution $P$ minimizing

$$
\min_{P: P_0 = p_{\mathrm{src}}}
    \mathcal{D}_{\mathrm{KL}}(P \,\|\, R),
$$

subject to the single constraint $P_0 = p_{\mathrm{src}}$. Similarly, the *backward half-bridge* constrains only the terminal distribution,

$$
\min_{P: P_T = p_{\mathrm{tgt}}}
    \mathcal{D}_{\mathrm{KL}}(P \,\|\, R).
$$

In words, the forward half-bridge asks: among all processes starting from the desired initial distribution, which one looks most like the reference? The backward half-bridge asks the same question for processes ending at the desired terminal distribution. By combining these two relaxations iteratively, one can approximate the full SB.

**Diffusion Models Miss Exact Endpoint Matching.** A key difference between diffusion models and the SB framework lies in the treatment of the terminal distribution $p_T$. In standard diffusion models, the forward SDE is typically linear in ${\mathbf{x}}_t$ (see [Equation (4.4.2)](../04-score-sde-framework/04-sde-instantiations.md#eq-forward-linear-sde)) and designed so that $p_T$ approximates the prior only as $T \to \infty$:

$$
p_T \approx p_{\mathrm{prior}}.
$$

At finite time, however, $p_T$ is the distribution of $\alpha_T{\mathbf{x}}_0+\sigma_T{\boldsymbol{\epsilon}}$ and therefore generally still depends on $p_{\mathrm{data}}$; it need not coincide exactly with the prescribed Gaussian prior (see [Section C.1.5](../../appendices/c-ito-calculus/01-ito-formula.md#subsec-rigorous-proof-linear-sde)).

In contrast, the SB framework enforces exact marginal matching at a finite time $T$ by introducing an additional control drift of the form $g^2(t)\nabla_\mathbf{x}\log \Psi(\mathbf{x}, t)$. This ensures that the terminal distribution precisely satisfies $p_T = p_{\mathrm{prior}}$, regardless of the initial data distribution $p_0 = p_{\mathrm{data}}$. In summary:

-   **Diffusion Models:** $p_T \approx p_{\mathrm{prior}}$, asymptotically as $T \to \infty$,

-   **Schrödinger Bridge:** $p_T = p_{\mathrm{prior}}$ exactly at finite $T$, enabled by solving for the control potentials $\Psi$ and $\widehat{\Psi}$.

**Diffusion Schrödinger Bridge.** The fixed forward noising process can be viewed as a forward Schrödinger half-bridge anchored at $p_{\mathrm{data}}$ relative to the chosen reference dynamics. Its terminal marginal is the distribution produced by that reference process and is generally only approximately $p_{\mathrm{prior}}$; hence it is not the full bridge between the prescribed pair $(p_{\mathrm{data}},p_{\mathrm{prior}})$.

To address this, the Diffusion Schrödinger Bridge (DSB) ([De Bortoli et al., 2021](references.md#ref-de2021diffusion)) alternates between matching both endpoint marginals by following the idea of the Iterative Proportional Fitting (IPF) algorithm, an alternating projection method. This extends diffusion models to solve the full SB problem as follows[^1]:

-   **Step 0: Reference Process.** Initialize with $P^{(0)} := R_{\text{fwd}}$, the reference forward SDE:

    $$
    \mathrm{d} \mathbf{x}_t = \mathbf{f}(\mathbf{x}_t, t)   \mathrm{d} t + g(t)   \mathrm{d} \mathbf{w}_t, \quad \mathbf{x}_0 \sim p_{\text{data}}.
    $$

    This ensures $P_0^{(0)} = p_{\text{data}}$, but typically $P_T^{(0)} \neq p_{\text{prior}}$.

-   **Step 1: Backward Pass.** Compute the process $P^{(1)}$ that matches $p_{\text{prior}}$ at time $T$ while staying close to $P^{(0)}$:

    $$
    P^{(1)} = \operatorname*{arg\,min}_{P : P_T = p_{\text{prior}}} \mathcal{D}_{\text{KL}}(P  \|  P^{(0)}).
    $$

    This is achieved via approximating the oracle score function with a neural network $\mathbf{s}_{\boldsymbol{\phi}^\times}$, which results in the reverse-time SDE:

    $$
    \mathrm{d} \mathbf{x}_t = \left[ \mathbf{f}(\mathbf{x}_t, t) - g^2(t) \mathbf{s}_{\boldsymbol{\phi}^\times}(\mathbf{x}_t, t) \right] \mathrm{d} t + g(t)  \mathrm{d} \bar{\mathbf{w}}_t,
    $$

    simulated backward from $\mathbf{x}_T \sim p_{\text{prior}}$.

-   **Iteration.** The process $P^{(1)}$ satisfies $P_T^{(1)} = p_{\text{prior}}$, but its initial marginal $P_0^{(1)}$ typically deviates from $p_{\text{data}}$. IPF addresses this by learning a forward SDE to adjust $P_0^{(1)}$ back to $p_{\text{data}}$, followed by another backward pass to enforce $p_{\text{prior}}$. This alternation continues, refining the process until convergence to the optimal bridge $P^*$, which satisfies both $P_0^* = p_{\text{data}}$ and $P_T^* = p_{\text{prior}}$. [De Bortoli et al. (2021)](references.md#ref-de2021diffusion) prove convergence under mild conditions.

The DSB formulation above is historically important and conceptually clear, but its practical implementation can be computationally demanding. The forward–backward projections are carried out iteratively, so that each IPF iteration requires solving a separate score-matching problem, while approximation errors may also accumulate across rounds. More broadly, a growing body of work seeks to alleviate these difficulties by developing Schrödinger bridge formulations with improved stability and scalability, and in some cases with reduced dependence on repeatedly fitting separate diffusion models throughout the iterative procedure. We do not discuss these developments in detail here, but mention them to orient interested readers to a broader literature on practical SB methods ([Shi et al., 2023](references.md#ref-shi2023dsbm); [Liu et al., 2024](references.md#ref-liu2024gsbm)).

[^1]: Although this description uses $p_{\mathrm{data}}$ and $p_{\mathrm{prior}}$, the DSB framework applies to any pair of endpoint distributions.
