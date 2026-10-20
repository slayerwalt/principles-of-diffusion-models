# D.2 Score-Based Perspective {#app-sec-score-proof}

## D.2.1 [Proposition 3.2.1](../../chapters/03-score-based-perspective/02-score-based-models.md#sm-trace): Tractable Score Matching via Integration by Parts {#app-sec-sm-trace}

***Proof.***

**Expanding $\mathcal{L}_{\text{SM}}(\boldsymbol{\phi})$.** Let us expand the squared difference inside the expectation:

$$
\begin{aligned}
\mathcal{L}_{\text{SM}}(\boldsymbol{\phi}) &= \frac{1}{2}\mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}({\mathbf{x}})} \left[ \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}})\right\rVert_2^2 - 2 \langle {\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}), {\mathbf{s}}({\mathbf{x}}) \rangle + \left\lVert{\mathbf{s}}({\mathbf{x}})\right\rVert_2^2 \right] \\
        &= \frac{1}{2}\mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}({\mathbf{x}})} \left[ \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}})\right\rVert_2^2 \right] - \mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}({\mathbf{x}})} \left[ \langle {\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}), {\mathbf{s}}({\mathbf{x}}) \rangle \right] \\
        &\quad + \frac{1}{2}\mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}({\mathbf{x}})} \left[ \left\lVert{\mathbf{s}}({\mathbf{x}})\right\rVert_2^2 \right].
\end{aligned}
$$

We now focus on the cross-product term:

$$
\mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}({\mathbf{x}})} \left[ \langle {\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}), {\mathbf{s}}({\mathbf{x}}) \rangle \right].
$$

Using the fact that

$$
\nabla_{{\mathbf{x}}} \log p_{\text{data}}({\mathbf{x}}) = \frac{\nabla_{{\mathbf{x}}}p_{\text{data}}({\mathbf{x}})}{p_{\text{data}}({\mathbf{x}})},
$$

and assuming $p_{\text{data}}({\mathbf{x}})$ is not zero (e.g., on its support), the cross-product term becomes:

$$
\begin{aligned}
\mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}({\mathbf{x}})} \left[ \langle {\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}), {\mathbf{s}}({\mathbf{x}}) \rangle \right]
             &= \int {\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}})^\top  \nabla_{{\mathbf{x}}} \log p_{\text{data}}({\mathbf{x}}) p_{\text{data}}({\mathbf{x}}) \mathop{}\!\mathrm{d} {\mathbf{x}} \\
             &= \int {\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}})^\top  \nabla_{{\mathbf{x}}}p_{\text{data}}({\mathbf{x}}) \mathop{}\!\mathrm{d} {\mathbf{x}} \\
             &= \sum_{i=1}^{D} \int  s_{\boldsymbol{\phi}}^{(i)}({\mathbf{x}}) \partial_{x_i}p_{\text{data}}({\mathbf{x}}) \mathop{}\!\mathrm{d} {\mathbf{x}},
\end{aligned}
$$

where $s_{\boldsymbol{\phi}}^{(i)}({\mathbf{x}})$ is the $i$-th component of the score function

$$
{\mathbf{s}}_{\boldsymbol{\phi}} = \left( s_{\boldsymbol{\phi}}^{(1)}, s_{\boldsymbol{\phi}}^{(2)}, \dots, s_{\boldsymbol{\phi}}^{(D)} \right).
$$

**Integration by Parts.** We use the following integration-by-parts formula ([Evans, 2010](references.md#ref-evans10)), which is derived from standard calculus:

Let $u, v$ be differentiable real-valued functions on a ball $\mathbb{B}(\boldsymbol{0}, R) \subset \mathbb{R}^D$ of radius $R>0$. Then for $i = 1, \ldots, D$, the formula holds:

$$
\int_{\mathbb{B}(\boldsymbol{0}, R)} u   \partial_{x_i} v   \mathop{}\!\mathrm{d} {\mathbf{x}} = -\int_{\mathbb{B}(\boldsymbol{0}, R)} v   \partial_{x_i} u   \mathop{}\!\mathrm{d} {\mathbf{x}} + \int_{\partial \mathbb{B}(\boldsymbol{0}, R)} u v \nu_i   \mathop{}\!\mathrm{d} S,
$$

where $\nu = (\nu_1, \ldots, \nu_D)$ is the outward unit normal to the boundary $\partial \mathbb{B}(\boldsymbol{0}, R)$—a sphere with radius $R>0$, and $\mathop{}\!\mathrm{d} S$ is the surface measure on $\partial \mathbb{B}(\boldsymbol{0}, R)$.

We apply this formula to $u({\mathbf{x}}) := s_{\boldsymbol{\phi}}^{(i)}({\mathbf{x}})$ and $v({\mathbf{x}}) = p_{\text{data}}({\mathbf{x}})$ for all $i = 1, \dots, D$, assuming that

$$
|u({\mathbf{x}}) v({\mathbf{x}})| \to 0 \quad \text{as } R \to \infty.
$$

Summing the results over all $i = 1, \dots, D$, we get:

$$
\begin{aligned}
\mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}({\mathbf{x}})} \left[ \langle {\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}), {\mathbf{s}}({\mathbf{x}}) \rangle \right]
             &= - \sum_{i=1}^{D} \int  \partial_{x_i}s_{\boldsymbol{\phi}}^{(i)}({\mathbf{x}}) p_{\text{data}}({\mathbf{x}}) \mathop{}\!\mathrm{d} {\mathbf{x}} \\
             &= - \mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}({\mathbf{x}})} \left[\operatorname{Tr}\left(\nabla_{{\mathbf{x}}}{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}})\right) \right].
\end{aligned}
$$

Combining all results, we have:

$$
\begin{aligned}
\mathcal{L}_{\text{SM}}(\boldsymbol{\phi}) &= \underbrace{\mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}({\mathbf{x}})} \left[\operatorname{Tr}\left(\nabla_{{\mathbf{x}}}{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}})\right) + \frac{1}{2}  \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}})\right\rVert_2^2 \right]}_{\widetilde{\mathcal{L}}_{\text{SM}}(\boldsymbol{\phi})} \\
        &\quad + \underbrace{\frac{1}{2}\mathbb{E}_{{\mathbf{x}} \sim p_{\text{data}}({\mathbf{x}})} \left[ \left\lVert{\mathbf{s}}({\mathbf{x}})\right\rVert_2^2 \right]}_{=:C},
\end{aligned}
$$

where $C$ depends only on the distribution $p_{\text{data}}$, which concludes the proof. $\blacksquare$

## D.2.2 [Theorem 3.3.1](../../chapters/03-score-based-perspective/03-denoising-score-matching.md#thm-sm-dsm): Equivalence Between SM and DSM Minimization {#app-sec-sm-dsm}

***Proof.*** Expanding both $\mathcal{L}_{\text{SM}}(\boldsymbol{\phi}; \sigma)$ and $\mathcal{L}_{\text{DSM}}(\boldsymbol{\phi}; \sigma)$, we have:

$$
\begin{aligned}
&\mathcal{L}_{\text{SM}}(\boldsymbol{\phi}; \sigma) = \frac{1}{2} \mathbb{E}_{\tilde{{\mathbf{x}}} \sim p_\sigma(\tilde{{\mathbf{x}}})}
 \begin{aligned}[t]
         \Big[
    \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)\right\rVert_2^2
    &- 2 {\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)^\top  \nabla_{\tilde{{\mathbf{x}}}} \log p_\sigma(\tilde{{\mathbf{x}}}) \\
    &+ \left\lVert\nabla_{\tilde{{\mathbf{x}}}} \log p_\sigma(\tilde{{\mathbf{x}}})\right\rVert_2^2
    \Big],
 \end{aligned}
\\
&\mathcal{L}_{\text{DSM}}(\boldsymbol{\phi}; \sigma) = \frac{1}{2} \mathbb{E}_{ p_{\text{data}}({\mathbf{x}})p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}})}
 \begin{aligned}[t]
         \Big[
    \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)\right\rVert_2^2
    &- 2 {\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)^\top \nabla_{\tilde{{\mathbf{x}}}} \log p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}}) \\
    &+ \left\lVert\nabla_{\tilde{{\mathbf{x}}}} \log p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}})\right\rVert_2^2
    \Big].
 \end{aligned}
\end{aligned}
$$

Subtracting the two equations yields:

$$
\begin{aligned}
&\mathcal{L}_{\text{SM}}(\boldsymbol{\phi}; \sigma) -\mathcal{L}_{\text{DSM}}(\boldsymbol{\phi}; \sigma)
 \\= \frac{1}{2} &\Bigg( \mathbb{E}_{\tilde{{\mathbf{x}}} \sim p_\sigma(\tilde{{\mathbf{x}}})} \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)\right\rVert_2^2 - \mathbb{E}_{ p_{\text{data}}({\mathbf{x}})p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}})} \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)\right\rVert_2^2\Bigg)
 \\ \quad - &
 \begin{aligned}[t]
 \Bigg(
      \mathbb{E}_{\tilde{{\mathbf{x}}} \sim p_\sigma(\tilde{{\mathbf{x}}})}  &\left[ {\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)^\top  \nabla_{\tilde{{\mathbf{x}}}} \log p_\sigma(\tilde{{\mathbf{x}}}) \right] \\
      &- \mathbb{E}_{ p_{\text{data}}({\mathbf{x}})p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}})}\left[ {\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)^\top  \nabla_{\tilde{{\mathbf{x}}}} \log p_\sigma(\tilde{{\mathbf{x}}}| {\mathbf{x}})\right]
      \Bigg)
 \end{aligned}
 \\ \quad +  \frac{1}{2}&\Bigg(\mathbb{E}_{\tilde{{\mathbf{x}}} \sim p_\sigma(\tilde{{\mathbf{x}}})}  \left\lVert\nabla_{\tilde{{\mathbf{x}}}} \log p_\sigma(\tilde{{\mathbf{x}}})\right\rVert_2^2  -  \mathbb{E}_{ p_{\text{data}}({\mathbf{x}})p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}})} \left\lVert\nabla_{\tilde{{\mathbf{x}}}} \log p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}})\right\rVert_2^2   \Bigg).
\end{aligned}
$$

Next, we address the three terms one at a time. For the first term, since $p_\sigma(\tilde{{\mathbf{x}}}) = \int p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}}) p_{\text{data}}({\mathbf{x}}) \mathop{}\!\mathrm{d}{\mathbf{x}}$, we can rewrite it as:

$$
\begin{aligned}
\mathbb{E}_{\tilde{{\mathbf{x}}} \sim p_\sigma(\tilde{{\mathbf{x}}})} \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)\right\rVert_2^2
    &= \int \Big( \int p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}}) p_{\text{data}}({\mathbf{x}})  \mathop{}\!\mathrm{d}{\mathbf{x}} \Big) \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)\right\rVert_2^2 \mathop{}\!\mathrm{d} \tilde{{\mathbf{x}}}
    \\ &= \int p_{\text{data}}({\mathbf{x}}) \int p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}}) \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)\right\rVert_2^2 \mathop{}\!\mathrm{d} \tilde{{\mathbf{x}}} \mathop{}\!\mathrm{d}{\mathbf{x}}
    \\ &= \mathbb{E}_{ p_{\text{data}}({\mathbf{x}})p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}})} \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)\right\rVert_2^2.
\end{aligned}
$$

Thus, the first term is zero. For the second term:

<span id="eq-dsm-sec-term"></span>

$$
\begin{aligned}
          &\mathbb{E}_{\tilde{{\mathbf{x}}} \sim p_\sigma(\tilde{{\mathbf{x}}})}\left[ {\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)^\top  \nabla_{\tilde{{\mathbf{x}}}} \log p_\sigma(\tilde{{\mathbf{x}}}) \right]
      \\=&\int  p_\sigma(\tilde{{\mathbf{x}}}) {\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)^\top  \frac{\nabla_{\tilde{{\mathbf{x}}}} p_\sigma(\tilde{{\mathbf{x}}})}{p_\sigma(\tilde{{\mathbf{x}}})}  \mathop{}\!\mathrm{d}\tilde{\mathbf{x}}
        \\=&\int   {\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)^\top  \nabla_{\tilde{{\mathbf{x}}}} \int p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}}) p_{\text{data}}({\mathbf{x}}) \mathop{}\!\mathrm{d}{\mathbf{x}} \mathop{}\!\mathrm{d}\tilde{\mathbf{x}}
    \\=&\int \int  {\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)^\top    \nabla_{\tilde{{\mathbf{x}}}}p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}}) p_{\text{data}}({\mathbf{x}}) \mathop{}\!\mathrm{d}\tilde{\mathbf{x}} \mathop{}\!\mathrm{d}{\mathbf{x}}
      \\=  &\mathbb{E}_{ p_{\text{data}}({\mathbf{x}})p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}})}\left[ {\mathbf{s}}_{\boldsymbol{\phi}}(\tilde{{\mathbf{x}}}; \sigma)^\top  \nabla_{\tilde{{\mathbf{x}}}} \log p_\sigma(\tilde{{\mathbf{x}}}| {\mathbf{x}})\right].
\end{aligned}
\tag{D.2.1}
$$

Thus, it is also zero. For the third term, note that:

$$
C:=\frac{1}{2}\Bigg(\mathbb{E}_{\tilde{{\mathbf{x}}} \sim p_\sigma(\tilde{{\mathbf{x}}})}  \left\lVert\nabla_{\tilde{{\mathbf{x}}}} \log p_\sigma(\tilde{{\mathbf{x}}})\right\rVert_2^2  -  \mathbb{E}_{ p_{\text{data}}({\mathbf{x}})p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}})} \left\lVert\nabla_{\tilde{{\mathbf{x}}}} \log p_{\sigma}(\tilde{{\mathbf{x}}} | {\mathbf{x}})\right\rVert_2^2   \Bigg)
$$

depends only on $p_{\text{data}}({\mathbf{x}})$ and $p_{\sigma}(\tilde{{\mathbf{x}}}|{\mathbf{x}})$, and hence it is constant with respect to $\boldsymbol{\phi}$. $\blacksquare$

## D.2.3 [Lemma 3.3.2](../../chapters/03-score-based-perspective/03-denoising-score-matching.md#tweedie): Tweedie’s Formula {#app-sec-tweedie}

We first state a more general form of Tweedie’s formula, which considers time-dependent Gaussian perturbations, and we provide its proof below.

**Tweedie’s Identity with Time-Dependent Parameters.** Let ${\mathbf{x}}_t|{\mathbf{x}}_0 \sim \mathcal{N}\big(\cdot; \alpha_t{\mathbf{x}}_0, \sigma_t^2{\mathbf{I}}\big)$. Then Tweedie’s formula says

$$
\begin{aligned}
\alpha_t \mathbb{E}_{\mathbf{x}_0 \sim p(\mathbf{x}_0|{\mathbf{x}}_t)}[\mathbf{x}_0|{\mathbf{x}}_t] = {\mathbf{x}}_t + \sigma_t^2 \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t),
\end{aligned}
$$

where the expectation is taken over the posterior distribution $p(\mathbf{x}_0|{\mathbf{x}}_t)$ of $\mathbf{x}_0$ given the observed ${\mathbf{x}}_t$, and $p_t({\mathbf{x}}_t)$ is the marginal density of ${\mathbf{x}}_t$.

***Proof.***

**Marginal Density and Its Score.** We recall that the marginal density of ${\mathbf{x}}_t$ is given by

$$
\begin{aligned}
p_t({\mathbf{x}}_t) = \int p_t({\mathbf{x}}_t|{\mathbf{x}}_0)  p_0({\mathbf{x}}_0)  \mathop{}\!\mathrm{d}{\mathbf{x}}_0.
\end{aligned}
$$

We now compute the score function:

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t)
= \frac{\nabla_{{\mathbf{x}}_t} p_t({\mathbf{x}}_t)}{p_t({\mathbf{x}}_t)} = \frac{1}{p_t({\mathbf{x}}_t)} \int \nabla_{{\mathbf{x}}_t} p_t({\mathbf{x}}_t|{\mathbf{x}}_0)  p_0({\mathbf{x}}_0)  \mathop{}\!\mathrm{d}{\mathbf{x}}_0.
\end{aligned}
$$

We therefore need to compute the gradient of the conditional density.

**Gradient of the Conditional and Rearrangement.** The gradient of the conditional Gaussian density is:

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t} p_t({\mathbf{x}}_t|{\mathbf{x}}_0)
= -p_t({\mathbf{x}}_t|{\mathbf{x}}_0) \cdot \sigma_t^{-2} ({\mathbf{x}}_t - \alpha_t {\mathbf{x}}_0).
\end{aligned}
$$

Substituting this into the previous expression, we have:

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t} p_t({\mathbf{x}}_t)
&= \int \nabla_{{\mathbf{x}}_t} p_t({\mathbf{x}}_t|{\mathbf{x}}_0)  p_0({\mathbf{x}}_0)  \mathop{}\!\mathrm{d}{\mathbf{x}}_0 \\
&= -\sigma_t^{-2} \int ({\mathbf{x}}_t - \alpha_t {\mathbf{x}}_0)  p_t({\mathbf{x}}_t|{\mathbf{x}}_0)  p_0({\mathbf{x}}_0)  \mathop{}\!\mathrm{d}{\mathbf{x}}_0 \\
&= -\sigma_t^{-2} \int ({\mathbf{x}}_t - \alpha_t {\mathbf{x}}_0)  p({\mathbf{x}}_0|{\mathbf{x}}_t)  p_t({\mathbf{x}}_t)  \mathop{}\!\mathrm{d}{\mathbf{x}}_0 \\
&= -p_t({\mathbf{x}}_t)  \sigma_t^{-2} \left({\mathbf{x}}_t - \alpha_t  \mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_t)}[{\mathbf{x}}_0\vert {\mathbf{x}}_t] \right).
\end{aligned}
$$

Dividing both sides by $p_t({\mathbf{x}}_t)$, we obtain:

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t)
= -\sigma_t^{-2} \left({\mathbf{x}}_t - \alpha_t  \mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_t)}[{\mathbf{x}}_0\vert {\mathbf{x}}_t] \right).
\end{aligned}
$$

Rearranging yields:

$$
\begin{aligned}
{\mathbf{x}}_t + \sigma_t^{2} \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t)
= \alpha_t  \mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_t)}[{\mathbf{x}}_0\vert {\mathbf{x}}_t].
\end{aligned}
$$

This completes the derivation.

## D.2.4 Stein’s Identity and Surrogate SURE Objective {#app-sec-stein-identity}

**Stein’s Identity.** Stein’s identity is the integration-by-parts technique that turns expectations under an unknown density into expectations of observable functions and their derivatives, which cancels the partition function and enables unbiased, tractable objectives and tests without ever evaluating the unknown density or the partition function. We begin with the simplest one-dimensional case and then extend it to the form needed to prove the surrogate loss for SURE.

**1D, Standard Normal Case.** If $z \sim \mathcal{N}(0,1)$ and $f$ has suitable decay, then Stein’s identity states:

$$
{\mathbb{E}}[f'(z)]  =  {\mathbb{E}}[z f(z)] .
$$

Denote $\phi(z) := \tfrac{1}{\sqrt{2\pi}} e^{-z^2/2}$, the one-dimensional standard normal density. The proof follows by integration by parts, using $\phi'(z) = -z\phi(z)$, together with the vanishing boundary term. To see this precisely, we compute

$$
{\mathbb{E}}[f'(z)]
= \int_{-\infty}^{\infty} f'(z)  \phi(z)  \mathop{}\!\mathrm{d} z.
$$

By integration by parts, with $u=\phi(z)$ and $\mathop{}\!\mathrm{d} v=f'(z) \mathop{}\!\mathrm{d} z$, we obtain

$$
\int f'(z)  \phi(z)  \mathop{}\!\mathrm{d} z
= \Big[ f(z)  \phi(z) \Big]_{-\infty}^{\infty}
- \int f(z)  \phi'(z)  \mathop{}\!\mathrm{d} z.
$$

Since $\phi'(z) = -z\phi(z)$ and $f(z)\phi(z)\to 0$ as $|z|\to\infty$ (decay condition), the boundary term vanishes and we have

$$
{\mathbb{E}}[f'(z)]  =  \int f(z)  z  \phi(z)  \mathop{}\!\mathrm{d} z  =  {\mathbb{E}}[z f(z)].
$$

This completes the derivation and proves the one-dimensional case of Stein’s identity.

**Multivariate, Standard Normal Case.** If ${\mathbf{z}} \sim \mathcal{N}(\mathbf{0},{\mathbf{I}}_D)$ and $g:{\mathbb{R}}^D \to {\mathbb{R}}$, then Stein’s identity is

$$
{\mathbb{E}}[\nabla g({\mathbf{z}})]  =  {\mathbb{E}}[{\mathbf{z}} g({\mathbf{z}})].
$$

Equivalently, for $\mathbf{u}:{\mathbb{R}}^D \to {\mathbb{R}}^D$,

<span id="eq-stein-multi"></span>

$$
\begin{aligned}
{\mathbb{E}}[ \nabla_{{\mathbf{z}}} \cdot\mathbf{u}({\mathbf{z}})]  =  {\mathbb{E}}[{\mathbf{z}}^\top \mathbf{u}({\mathbf{z}})] .
\end{aligned}
\tag{D.2.2}
$$

**Identity Needed for SURE.** With $\tilde{{\mathbf{x}}}={\mathbf{x}}+\sigma{\mathbf{z}}$, where ${\mathbf{z}} \sim \mathcal{N}(\mathbf{0},{\mathbf{I}}_D)$, and any vector function $\mathbf{a}$ of suitable regularity,

<span id="eq-stein-for-sure"></span>

$$
\begin{aligned}
{\mathbb{E}}\big[(\tilde{{\mathbf{x}}}-{\mathbf{x}})^\top \mathbf{a}(\tilde{{\mathbf{x}}})  \big|  {\mathbf{x}}\big]
 =  \sigma^2 {\mathbb{E}}\big[ \nabla_{\tilde{{\mathbf{x}}}} \cdot\mathbf{a}(\tilde{{\mathbf{x}}})  \big|  {\mathbf{x}}\big].
\end{aligned}
\tag{D.2.3}
$$

This is obtained by applying [Equation (D.2.2)](02-score-based-perspective.md#eq-stein-multi) and using the chain rule.

**Deriving SURE from the Conditional MSE.** Let ${\mathbf{D}}:{\mathbb{R}}^D \to {\mathbb{R}}^D$ be a denoiser and define

$$
R({\mathbf{D}};{\mathbf{x}}):={\mathbb{E}} \left[\|{\mathbf{D}}(\tilde{{\mathbf{x}}})-{\mathbf{x}}\|_2^2| {\mathbf{x}}\right].
$$

Expand around $\tilde{{\mathbf{x}}}$:

$$
\begin{aligned}
&~R({\mathbf{D}};{\mathbf{x}}) \\
=&~{\mathbb{E}} \left[\|{\mathbf{D}}(\tilde{{\mathbf{x}}})-\tilde{{\mathbf{x}}}\|^2  \big|  {\mathbf{x}}\right]
+2 {\mathbb{E}} \left[({\mathbf{D}}(\tilde{{\mathbf{x}}})-\tilde{{\mathbf{x}}})^\top(\tilde{{\mathbf{x}}}-{\mathbf{x}})  \big|  {\mathbf{x}}\right]
+{\mathbb{E}} \left[\|\tilde{{\mathbf{x}}}-{\mathbf{x}}\|^2  \big|  {\mathbf{x}}\right] \\
=&~{\mathbb{E}} \left[\|{\mathbf{D}}(\tilde{{\mathbf{x}}})-\tilde{{\mathbf{x}}}\|^2  \big|  {\mathbf{x}}\right]
+2\Big(\underbrace{{\mathbb{E}}[(\tilde{{\mathbf{x}}}-{\mathbf{x}})^\top {\mathbf{D}}(\tilde{{\mathbf{x}}})|{\mathbf{x}}]}_{\substack{\sigma^2 {\mathbb{E}}[ \nabla_{\tilde{{\mathbf{x}}}} \cdot{\mathbf{D}}(\tilde{{\mathbf{x}}})|{\mathbf{x}}] \\  \text{ by }\href{../02-score-based-perspective/#eq-stein-for-sure}{\text{Equation (D.2.3)}}
}}
-\underbrace{{\mathbb{E}}[(\tilde{{\mathbf{x}}}-{\mathbf{x}})^\top \tilde{{\mathbf{x}}}|{\mathbf{x}}]}_{ \sigma^2 D}\Big)
\\&\qquad\qquad\qquad+\underbrace{{\mathbb{E}}[\|\tilde{{\mathbf{x}}}-{\mathbf{x}}\|^2|{\mathbf{x}}]}_{ \sigma^2 D} \\
=&~{\mathbb{E}} \left[\|{\mathbf{D}}(\tilde{{\mathbf{x}}})-\tilde{{\mathbf{x}}}\|^2
+2\sigma^2  \nabla_{\tilde{{\mathbf{x}}}} \cdot{\mathbf{D}}(\tilde{{\mathbf{x}}})
-D\sigma^2 | {\mathbf{x}}\right].
\end{aligned}
$$

Therefore the *observable* surrogate

$$
\mathrm{SURE}({\mathbf{D}};\tilde{{\mathbf{x}}})
:= \|{\mathbf{D}}(\tilde{{\mathbf{x}}})-\tilde{{\mathbf{x}}}\|_2^2
+2\sigma^2  \nabla_{\tilde{{\mathbf{x}}}} \cdot{\mathbf{D}}(\tilde{{\mathbf{x}}})
-D\sigma^2
$$

satisfies ${\mathbb{E}} \left[\mathrm{SURE}({\mathbf{D}};\tilde{{\mathbf{x}}}) \big| {\mathbf{x}}\right]=R({\mathbf{D}};{\mathbf{x}})$. Minimizing SURE (in expectation or empirically) is thus equivalent to minimizing the true conditional MSE while using only noisy observations.

## D.2.5 [Theorem 4.2.1](../../chapters/04-score-sde-framework/02-matching-marginals.md#thm-fpe): Marginal Alignment via Fokker–Planck {#app-sec-scoresde-fpe-proof}

***Proof.***

**Part 1: Fokker-Planck Equation for the Forward SDE.**

Consider the forward SDE:

$$
\mathop{}\!\mathrm{d} {\mathbf{x}}(t) = {\mathbf{f}}({\mathbf{x}}(t), t)  \mathop{}\!\mathrm{d} t + g(t)  \mathop{}\!\mathrm{d} {\mathbf{w}}(t).
$$

The diffusion matrix is $\sigma(t) = g(t) I_D$, so $\sigma(t) \sigma(t)^T = g^2(t) I_D$. The Fokker-Planck equation for the marginal density $p_t({\mathbf{x}})$ of ${\mathbf{x}}(t)$ is:

$$
\partial_t p_t({\mathbf{x}}) = -\nabla_{{\mathbf{x}}} \cdot \bigl[ {\mathbf{f}}({\mathbf{x}}, t) p_t({\mathbf{x}}) \bigr] + \frac{1}{2} \sum_{i,j=1}^D \frac{\partial^2}{\partial x_i \partial x_j} \bigl[ (g^2(t) \delta_{ij}) p_t({\mathbf{x}}) \bigr].
$$

Compute the diffusion term:

$$
\sum_{i,j=1}^D \frac{\partial^2}{\partial x_i \partial x_j} \bigl[ g^2(t) \delta_{ij} p_t({\mathbf{x}}) \bigr] = \sum_{i=1}^D \frac{\partial^2}{\partial x_i^2} \bigl[ g^2(t) p_t({\mathbf{x}}) \bigr] = g^2(t) \Delta_{{\mathbf{x}}} p_t({\mathbf{x}}).
$$

Thus:

$$
\partial_t p_t({\mathbf{x}}) = -\nabla_{{\mathbf{x}}} \cdot \bigl[ {\mathbf{f}}({\mathbf{x}}, t) p_t({\mathbf{x}}) \bigr] + \frac{1}{2} g^2(t) \Delta_{{\mathbf{x}}} p_t({\mathbf{x}}).
$$

Now, rewrite using:

$$
{\mathbf{v}}({\mathbf{x}}, t) = {\mathbf{f}}({\mathbf{x}}, t) - \frac{1}{2} g^2(t) \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}).
$$

Since $\nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}) = \frac{\nabla_{{\mathbf{x}}} p_t({\mathbf{x}})}{p_t({\mathbf{x}})}$, compute:

$$
\nabla_{{\mathbf{x}}} \cdot \bigl[ {\mathbf{v}}({\mathbf{x}}, t) p_t({\mathbf{x}}) \bigr] = \nabla_{{\mathbf{x}}} \cdot \biggl[ {\mathbf{f}}({\mathbf{x}}, t) p_t({\mathbf{x}}) - \frac{1}{2} g^2(t) \frac{\nabla_{{\mathbf{x}}} p_t({\mathbf{x}})}{p_t({\mathbf{x}})} p_t({\mathbf{x}}) \biggr].
$$

The second term is:

$$
\nabla_{{\mathbf{x}}} \cdot \biggl[ -\frac{1}{2} g^2(t) \nabla_{{\mathbf{x}}} p_t({\mathbf{x}}) \biggr] = -\frac{1}{2} g^2(t) \Delta_{{\mathbf{x}}} p_t({\mathbf{x}}).
$$

Thus:

$$
\nabla_{{\mathbf{x}}} \cdot \bigl[ {\mathbf{v}}({\mathbf{x}}, t) p_t({\mathbf{x}}) \bigr] = \nabla_{{\mathbf{x}}} \cdot \bigl[ {\mathbf{f}}({\mathbf{x}}, t) p_t({\mathbf{x}}) \bigr] - \frac{1}{2} g^2(t) \Delta_{{\mathbf{x}}} p_t({\mathbf{x}}).
$$

Therefore:

$$
\partial_t p_t({\mathbf{x}}) = -\nabla_{{\mathbf{x}}} \cdot \bigl[ {\mathbf{v}}({\mathbf{x}}, t) p_t({\mathbf{x}}) \bigr],
$$

verifying the Fokker-Planck equation in both forms.

**Part 2: PF-ODE Marginal Densities.**

Consider the PF-ODE:

$$
\frac{\mathop{}\!\mathrm{d} \tilde{\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t} = {\mathbf{v}}(\tilde{\mathbf{x}}(t), t), \quad {\mathbf{v}}({\mathbf{x}}, t) = {\mathbf{f}}({\mathbf{x}}, t) - \frac{1}{2} g^2(t) \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}).
$$

**Forward Direction: $\tilde{\mathbf{x}}(0) \sim p_0$.** Let $\tilde{\mathbf{x}}(t)$ follow the PF-ODE with $\tilde{\mathbf{x}}(0) \sim p_0$. The flow map $\boldsymbol{\Psi}_t: \mathbb{R}^D \to \mathbb{R}^D$ is defined by:

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} \boldsymbol{\Psi}_t(\mathbf{x}_0) = {\mathbf{v}}(\boldsymbol{\Psi}_t(\mathbf{x}_0), t), \quad \boldsymbol{\Psi}_0(\mathbf{x}_0) = \mathbf{x}_0.
$$

Since $\tilde{\mathbf{x}}(t) = \boldsymbol{\Psi}_t(\tilde{\mathbf{x}}(0))$, the density $\tilde p_t({\mathbf{x}})$ of $\tilde{\mathbf{x}}(t)$ satisfies the continuity equation:

$$
\partial_t \tilde p_t({\mathbf{x}}) = -\nabla_{{\mathbf{x}}} \cdot \bigl[ {\mathbf{v}}({\mathbf{x}}, t) \tilde p_t({\mathbf{x}}) \bigr].
$$

Since $\tilde{\mathbf{x}}(0) \sim p_0$, we have $\tilde p_0({\mathbf{x}}) = p_0({\mathbf{x}})$. From Part 1, $p_t({\mathbf{x}})$ satisfies:

$$
\partial_t p_t({\mathbf{x}}) = -\nabla_{{\mathbf{x}}} \cdot \bigl[ {\mathbf{v}}({\mathbf{x}}, t) p_t({\mathbf{x}}) \bigr].
$$

Both $\tilde p_t$ and $p_t$ satisfy the same continuity equation with the same initial condition $p_0$. Assuming sufficient smoothness (e.g., ${\mathbf{v}} \in C^1$), the solution is unique in some appropriate function space, so $\tilde p_t = p_t$. Thus, $\tilde{\mathbf{x}}(t) \sim p_t$ for all $t \in [0, T]$.

**Backward Direction: $\tilde{\mathbf{x}}(T) \sim p_T$.** Now, let $\tilde{\mathbf{x}}(t)$ follow the PF-ODE backward from $t = T$ to $t = 0$, with $\tilde{\mathbf{x}}(T) \sim p_T$. The ODE is:

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} \tilde{\mathbf{x}}(t) = {\mathbf{v}}(\tilde{\mathbf{x}}(t), t).
$$

Let $s = T - t$, so the backward ODE becomes:

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s} \tilde{\mathbf{x}}(T - s) = -{\mathbf{v}}(\tilde{\mathbf{x}}(T - s), T - s).
$$

The density $\tilde p_{T-s}({\mathbf{x}})$ of $\tilde{\mathbf{x}}(T - s)$ satisfies:

$$
\partial_s \tilde p_{T-s}({\mathbf{x}}) = \nabla_{{\mathbf{x}}} \cdot \bigl[ {\mathbf{v}}({\mathbf{x}}, T - s) \tilde p_{T-s}({\mathbf{x}}) \bigr].
$$

Since $\tilde{\mathbf{x}}(T) \sim p_T$, we have $\tilde p_T = p_T$. The Fokker-Planck equation for $p_t$ at $t = T - s$ is:

$$
\partial_t p_{T-s}({\mathbf{x}}) = -\nabla_{{\mathbf{x}}} \cdot \bigl[ {\mathbf{v}}({\mathbf{x}}, T - s) p_{T-s}({\mathbf{x}}) \bigr].
$$

Since $\partial_t = -\partial_s$, we get:

$$
\partial_s p_{T-s}({\mathbf{x}}) = \nabla_{{\mathbf{x}}} \cdot \bigl[ {\mathbf{v}}({\mathbf{x}}, T - s) p_{T-s}({\mathbf{x}}) \bigr].
$$

Both $\tilde p_{T-s}$ and $p_{T-s}$ satisfy the same PDE with the same initial condition at $s = 0$ ($\tilde p_T = p_T$). Uniqueness implies $\tilde p_{T-s} = p_{T-s}$, so $\tilde{\mathbf{x}}(t) = \tilde{\mathbf{x}}(T - s) \sim p_{T-s} = p_t$, for all $t \in [0, T]$.

**Part 3: Reverse-Time SDE Marginal Densities.** We verify the marginal distributions using the forward-running reverse clock

$$
s:=T-t.
$$

Define

$$
{\mathbf{y}}(s):=\bar{\mathbf{x}}(T-s),
\qquad s\in[0,T].
$$

Since the reverse process starts from $\bar{\mathbf{x}}(T)\sim p_T$, we have

$$
{\mathbf{y}}(0)\sim p_T.
$$

Under this change of clock, the reverse-time SDE becomes

$$
\mathop{}\!\mathrm{d}{\mathbf{y}}(s)
=
\Bigl[
-{\mathbf{f}}({\mathbf{y}}(s),T-s)
+
g^2(T-s)\nabla_{{\mathbf{x}}}\log p_{T-s}({\mathbf{y}}(s))
\Bigr]\mathop{}\!\mathrm{d} s
+
g(T-s)\mathop{}\!\mathrm{d}{\mathbf{w}}^{\mathrm{rev}}(s),
$$

where ${\mathbf{w}}^{\mathrm{rev}}$ is a standard Wiener process under the forward-running clock $s$.

Let $q_s$ denote the density of ${\mathbf{y}}(s)$. Its Fokker–Planck equation is

$$
\begin{aligned}
\partial_s q_s({\mathbf{x}})
&=
-\nabla_{{\mathbf{x}}}\cdot
\Bigl(
\bigl[
-{\mathbf{f}}({\mathbf{x}},T-s)
+
g^2(T-s)\nabla_{{\mathbf{x}}}\log p_{T-s}({\mathbf{x}})
\bigr]
q_s({\mathbf{x}})
\Bigr)
\\
&\qquad
+
\tfrac12 g^2(T-s)\Delta_{{\mathbf{x}}}q_s({\mathbf{x}}).
\end{aligned}
$$

We now check that

$$
q_s({\mathbf{x}})=p_{T-s}({\mathbf{x}})
$$

satisfies this equation. Using

$$
\nabla_{{\mathbf{x}}}\log p_{T-s}({\mathbf{x}})\,p_{T-s}({\mathbf{x}})
=
\nabla_{{\mathbf{x}}}p_{T-s}({\mathbf{x}}),
$$

the right-hand side becomes

$$
\begin{aligned}
&\nabla_{{\mathbf{x}}}\cdot
\bigl[
{\mathbf{f}}({\mathbf{x}},T-s)p_{T-s}({\mathbf{x}})
\bigr]
-
g^2(T-s)\Delta_{{\mathbf{x}}}p_{T-s}({\mathbf{x}})
\\
&\qquad
+
\tfrac12 g^2(T-s)\Delta_{{\mathbf{x}}}p_{T-s}({\mathbf{x}})
\\
&=
\nabla_{{\mathbf{x}}}\cdot
\bigl[
{\mathbf{f}}({\mathbf{x}},T-s)p_{T-s}({\mathbf{x}})
\bigr]
-
\tfrac12 g^2(T-s)\Delta_{{\mathbf{x}}}p_{T-s}({\mathbf{x}}).
\end{aligned}
$$

On the other hand, the forward Fokker–Planck equation gives

$$
\partial_t p_t({\mathbf{x}})
=
-\nabla_{{\mathbf{x}}}\cdot
\bigl[{\mathbf{f}}({\mathbf{x}},t)p_t({\mathbf{x}})\bigr]
+
\tfrac12g^2(t)\Delta_{{\mathbf{x}}}p_t({\mathbf{x}}).
$$

Since $t=T-s$, the chain rule gives

$$
\begin{aligned}
\partial_s p_{T-s}({\mathbf{x}})
&=
-\partial_t p_t({\mathbf{x}})\big|_{t=T-s}
\\
&=
\nabla_{{\mathbf{x}}}\cdot
\bigl[
{\mathbf{f}}({\mathbf{x}},T-s)p_{T-s}({\mathbf{x}})
\bigr]
-
\tfrac12g^2(T-s)\Delta_{{\mathbf{x}}}p_{T-s}({\mathbf{x}}).
\end{aligned}
$$

This is exactly the Fokker–Planck equation of the reversed process.

Finally, $q_0=p_T$, which matches the initial distribution ${\mathbf{y}}(0)\sim p_T$. Under the regularity assumptions ensuring uniqueness of the Fokker–Planck solution,

$$
q_s=p_{T-s},
\qquad s\in[0,T].
$$

Returning to the original diffusion clock $t=T-s$, we obtain

$$
\bar{\mathbf{x}}(t)\sim p_t.
$$

Thus, when the reverse-time SDE is solved backward from $t=T$ to $t=0$, it follows the same marginal distributions as the forward SDE, but in the reverse order. $\blacksquare$

## D.2.6 [Proposition 4.3.1](../../chapters/04-score-sde-framework/03-training-and-sampling.md#dsm-minimizer): Minimizer of SM and DSM {#app-sec-min-dsm}

***Proof.*** To find the minimizer $\mathbf{s}^*$, we first consider a fixed time $t$ and analyze the inner expectation in the objective function:

$$
\begin{aligned}
\mathcal  J(t, \boldsymbol{\phi})
&:= \mathbb{E}_{{\mathbf{x}}_0 \sim p_{\text{data}}}
\mathbb{E}_{{\mathbf{x}}_t \sim p_t(\cdot|{\mathbf{x}}_0)}
\left[\left\| \mathbf{s}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t) - \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t|{\mathbf{x}}_0) \right\|_2^2 \right].
\end{aligned}
$$

For this expectation to be minimized, we need to find $\mathbf{s}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t)$ that minimizes the expected squared error for each ${\mathbf{x}}_t$. We can rewrite this expectation using the joint distribution of $X_0$ and $X_t$:

$$
\begin{aligned}
\mathcal J(t, \boldsymbol{\phi})
= \iint p_{\text{data}}({\mathbf{x}}_0) p_t({\mathbf{x}}_t |{\mathbf{x}}_0)
\left\| \mathbf{s}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t) - \nabla_{{\mathbf{x}}_t} \log p({\mathbf{x}}_t |  {\mathbf{x}}_0) \right\|_2^2
 \mathop{}\!\mathrm{d}{\mathbf{x}}_0  \mathop{}\!\mathrm{d}{\mathbf{x}}_t.
\end{aligned}
$$

For each fixed ${\mathbf{x}}_t$, we need to minimize:

$$
\begin{aligned}
\int p({\mathbf{x}}_0 | X_t = {\mathbf{x}}_t) p_t( {\mathbf{x}}_t)
\left\| \mathbf{s}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t) - \nabla_{{\mathbf{x}}_t} \log p({\mathbf{x}}_t |  {\mathbf{x}}_0) \right\|_2^2
 \mathop{}\!\mathrm{d}{\mathbf{x}}_0.
\end{aligned}
$$

Since $p_t({\mathbf{x}}_t)$ is constant with respect to $\mathbf{s}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t)$, this is equivalent to minimizing:

$$
\begin{aligned}
\int p( {\mathbf{x}}_0 | X_t = {\mathbf{x}}_t)
\left\| \mathbf{s}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t) - \nabla_{{\mathbf{x}}_t} \log p({\mathbf{x}}_t |  {\mathbf{x}}_0) \right\|_2^2
  \mathop{}\!\mathrm{d}{\mathbf{x}}_0
\end{aligned}
$$

This is minimized when $\mathbf{s}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t)$ equals the conditional expectation:

$$
\begin{aligned}
\mathbf{s}^*({\mathbf{x}}_t, t)
= \mathbb{E}_{X_0 \sim p(X_0 | X_t = {\mathbf{x}}_t)}
\left[ \nabla_{{\mathbf{x}}_t} \log p({\mathbf{x}}_t | X_0) \right].
\end{aligned}
$$

Now we need to prove that this equals $\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t)$. By Bayes’ rule and the definition of marginal probability:

$$
\begin{aligned}
p_t({\mathbf{x}}_t)
= \int p_t({\mathbf{x}}_t | {\mathbf{x}}_0)
p_{\text{data}}({\mathbf{x}}_0)   \mathop{}\!\mathrm{d}{\mathbf{x}}_0.
\end{aligned}
$$

Taking the logarithm and then the gradient with respect to ${\mathbf{x}}_t$:

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t)
= \frac{\nabla_{{\mathbf{x}}_t} p_t({\mathbf{x}}_t)}{p_t({\mathbf{x}}_t)}
= \frac{\nabla_{{\mathbf{x}}_t} \int p_t({\mathbf{x}}_t | {\mathbf{x}}_0)
p_{\text{data}}({\mathbf{x}}_0)   \mathop{}\!\mathrm{d}{\mathbf{x}}_0}
{\int p_t({\mathbf{x}}_t | {\mathbf{x}}_0)
p_{\text{data}}({\mathbf{x}}_0)   \mathop{}\!\mathrm{d}{\mathbf{x}}_0}.
\end{aligned}
$$

Under suitable regularity conditions, we can exchange the gradient and integral:

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t)
= \frac{\int \nabla_{{\mathbf{x}}_t} p_t({\mathbf{x}}_t | {\mathbf{x}}_0)
p_{\text{data}}({\mathbf{x}}_0)   \mathop{}\!\mathrm{d}{\mathbf{x}}_0}
{\int p_t({\mathbf{x}}_t | {\mathbf{x}}_0)
p_{\text{data}}({\mathbf{x}}_0)   \mathop{}\!\mathrm{d}{\mathbf{x}}_0}.
\end{aligned}
$$

Using

$$
\nabla_{{\mathbf{x}}_t}p_t({\mathbf{x}}_t|{\mathbf{x}}_0)
=
p_t({\mathbf{x}}_t|{\mathbf{x}}_0)
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t|{\mathbf{x}}_0),
$$

and Bayes’ rule, we obtain

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t)
&=
\int
p({\mathbf{x}}_0|{\mathbf{x}}_t)
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t|{\mathbf{x}}_0)
\mathop{}\!\mathrm{d}{\mathbf{x}}_0 \\
&=
\mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_t)}
\left[
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t|{\mathbf{x}}_0)
\right]
=
\mathbf{s}^*({\mathbf{x}}_t,t),
\end{aligned}
$$

which completes the proof. $\blacksquare$

## D.2.7 Closed-Form Score Function of a Gaussian {#closed-form-gaussian-score}

For future reference, we summarize the formula for the score of a general multivariate normal distribution as the following lemma:

<span id="score-general-gaussian"></span>

**Lemma D.2.1: Score of Gaussian**

Let ${\mathbf{x}} \in \mathbb{R}^D$ and consider the multivariate normal distribution

$$
p(\tilde{{\mathbf{x}}} | {\mathbf{x}}) := \mathcal{N}(\tilde{{\mathbf{x}}}; \boldsymbol{\mu}, \boldsymbol{\Sigma}),
$$

where $\boldsymbol{\mu} \in \mathbb{R}^D$ is the mean and $\boldsymbol{\Sigma} \in \mathbb{R}^{D \times D}$ is an invertible covariance matrix. Its score function is

<span id="eq-score-gaussian-formula"></span>

$$
\begin{aligned}
\nabla_{\tilde{{\mathbf{x}}}} \log p(\tilde{{\mathbf{x}}} | {\mathbf{x}}) = -\boldsymbol{\Sigma}^{-1} (\tilde{{\mathbf{x}}} - \boldsymbol{\mu}).
\end{aligned}
\tag{D.2.4}
$$

**Proof for Lemma.**

The density function of $p(\tilde{{\mathbf{x}}} | {\mathbf{x}})$ is given by:

$$
p(\tilde{{\mathbf{x}}} | {\mathbf{x}}) = \frac{1}{(2\pi)^{D/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} (\tilde{{\mathbf{x}}} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\tilde{{\mathbf{x}}} - \boldsymbol{\mu}) \right).
$$

To compute the score function, we first take the log of $p(\tilde{{\mathbf{x}}} | {\mathbf{x}})$:

$$
\log p(\tilde{{\mathbf{x}}} | {\mathbf{x}}) = -\frac{D}{2} \log(2\pi) - \frac{1}{2} \log |\boldsymbol{\Sigma}| - \frac{1}{2} (\tilde{{\mathbf{x}}} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\tilde{{\mathbf{x}}} - \boldsymbol{\mu}).
$$

Now, we compute the gradient of $\log p(\tilde{{\mathbf{x}}} | {\mathbf{x}})$ with respect to $\tilde{{\mathbf{x}}}$:

$$
\nabla_{\tilde{{\mathbf{x}}}} \log p(\tilde{{\mathbf{x}}} | {\mathbf{x}}) = - \frac{1}{2} \nabla_{\tilde{{\mathbf{x}}}} \left( (\tilde{{\mathbf{x}}} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\tilde{{\mathbf{x}}} - \boldsymbol{\mu}) \right).
$$

Using the chain rule, we get:

$$
\nabla_{\tilde{{\mathbf{x}}}} \left( (\tilde{{\mathbf{x}}} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\tilde{{\mathbf{x}}} - \boldsymbol{\mu}) \right) = 2 \boldsymbol{\Sigma}^{-1} (\tilde{{\mathbf{x}}} - \boldsymbol{\mu}).
$$

Thus, the score function is:

<span id="eq-gaussian-score"></span>

$$
\begin{aligned}
\nabla_{\tilde{{\mathbf{x}}}} \log p(\tilde{{\mathbf{x}}} | {\mathbf{x}}) = - \boldsymbol{\Sigma}^{-1} (\tilde{{\mathbf{x}}} - \boldsymbol{\mu}).
\end{aligned}
\tag{D.2.5}
$$
