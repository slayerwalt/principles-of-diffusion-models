# 7.1 Prologue of Distribution-to-Distribution Translation {#sec-distribution-translation}

Diffusion models fix the terminal distribution to a standard Gaussian, $p_{\mathrm{prior}}$. However, many applications require *distribution-to-distribution* translation: transforming a source distribution $p_{\mathrm{src}}$ into a different target $p_{\mathrm{tgt}}$. Examples include converting sketches to photorealistic images or translating between artistic styles.

Modern diffusion methods provide practical ways to achieve this. *One-endpoint* methods such as SDEdit ([Meng et al., 2021](../../references.md#ref-meng2021sdedit)) begin with a source image at $t=0$, diffuse it to an intermediate step $t$, and then use a pre-trained diffusion model for the target domain to reverse the process. This produces an output that matches the style and content of the target distribution.

*Two-endpoint* methods, like Dual Diffusion Bridge ([Su et al., 2022](../../references.md#ref-su2022dual)), instead connect the two domains through a shared latent distribution, typically a Gaussian at $t=1$. A forward probability–flow ODE transports samples from $p_{\mathrm{src}}$ into this latent space, while a reverse ODE trained on the target domain maps them back to $p_{\mathrm{tgt}}$. Beyond such sampling-time approaches, the Flow Matching framework described in [Section 5.2](../05-flow-based-perspective/02-flow-matching-framework.md#sec-flow-matching-framework) offers a training-based alternative: it directly learns an ODE flow that continuously moves mass from $p_{\mathrm{src}}$ to $p_{\mathrm{tgt}}$.

Crucially, transforming between distributions requires more than two separately trained models. It demands a principled mapping that aligns the dynamics at *both* endpoints and does so in the “cheapest” (cost-efficient) way.

In this section, rather than surveying the many diffusion-based translation applications, we shift our focus to the mathematical foundations of this classic distribution-to-distribution problem. In particular, we highlight optimal transport (OT) and its entropic variant, the Schrödinger bridge (SB), which have long been studied in the theoretical community as canonical formulations of cost-efficient (in mathematical sense) distributional transformation.

At its core, the fundamental question is:

<span id="question-7-1-1"></span>

**Question 7.1.1**

Given two probability distributions, what is the most efficient way to transform one into the other while minimizing the total *cost*?

Here, the cost, $c(\mathbf{x}, \mathbf{y})$, is a non-negative function that assigns a penalty for moving a unit of mass from a point $\mathbf{x}$ to a point $\mathbf{y}$. A common choice is the squared distance, $c(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|^2$.

This section provides a brief overview to clarify how diffusion-based approaches, including flow matching, connect to classical and regularized optimal transport. The central question we aim to explore is:

<span id="question-7-1-2"></span>

**Question 7.1.2**

Is a diffusion model a form of optimal transport connecting $p_{\text{data}}$ and $p_{\text{prior}}$, and in what sense?

To address this question, we first clarify what “optimality” means in [Section 7.2](02-transport-problem-setups.md#sec-ot-intro). We review classical *optimal transport (OT)* in the static Monge–Kantorovich form ([Equation (7.2.1)](02-transport-problem-setups.md#eq-ot)) and its dynamic Benamou–Brenier formulation ([Equation (7.2.3)](02-transport-problem-setups.md#eq-bb-ot); minimizing kinetic energy subject to the continuity equation), as well as the entropy regularized variant (*entropic OT*) in [Equation (7.2.5)](02-transport-problem-setups.md#eq-eot-epsilon), which is equivalent to the *Schrödinger Bridge Problem* ([Equation (7.2.8)](02-transport-problem-setups.md#eq-sb-kinetic)). In the dynamic view, OT induces a deterministic flow satisfying the continuity equation, whereas SB induces a controlled diffusion whose marginals evolve by the Fokker-Planck equation. We provide a high level map between these formulations in [Section 7.3](03-relationships-between-formulations.md#sec-relation-ot).

We then split the discussion into two parts. First, in [Section 7.4](04-diffusion-sdes-and-schrodinger-bridges.md#sec-dm-and-sb), we explain that the fixed forward noising SDE used in standard diffusion models is not, by itself, a Schrödinger bridge between arbitrary $p_{\mathrm{src}}$ and $p_{\mathrm{tgt}}$: the forward process is a chosen reference diffusion and both forward-in-time or reverse-time SDE do not, in general, enforce exact endpoint matching to a prescribed target. Hence it is not entropic OT optimal unless one explicitly solves the SB problem with those endpoints; while it is an optimal solution to the half-bridge problem as it is anchored with one starting point.

Second, in [Section 7.5](05-probability-flow-odes-and-optimal-transport.md#sec-pfode-ot), we return to the generative setting with $p_{\mathrm{src}}=p_{\mathrm{prior}}$ (Gaussian) and $p_{\mathrm{tgt}}=p_{\mathrm{data}}$. The PF-ODE defines a deterministic map that transports $p_{\mathrm{prior}}$ to $p_{\mathrm{data}}$ by construction. However, this flow is generally not an OT map for a prescribed transport cost (e.g., quadratic $W_2$): it realizes one admissible deterministic coupling among many and does not minimize the Benamou–Brenier action. What follows is we discuss if the “rectify flow” procedure ([Section 5.4.1](../05-flow-based-perspective/04-canonical-affine-flow.md#subsec-rectify)) can lead to a OT map; however, in general, there is no such theoretical guarantee. Therefore, the exact characterization between diffusion model’s PF-ODE map and OT remains a challenging and unsolved problem.
