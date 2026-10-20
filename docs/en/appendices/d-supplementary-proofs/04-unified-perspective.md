# D.4 Theoretical Supplement: A Unified and Systematic View on Diffusion Models {#app-sec-equiv-para}

## D.4.1 [Proposition 6.3.1](../../chapters/06-unified-perspective/03-equivalent-parameterizations.md#equi-para): Equivalence of Parametrizations {#equivalence-of-parametrizations}

***Proof:*** As in [Theorem 4.3.1](../../chapters/04-score-sde-framework/03-training-and-sampling.md#dsm-minimizer) on the DSM loss, the global optimum of the matching loss

$$
\mathbb{E}_t\left[\omega(t)\,\mathbb{E}_{{\mathbf{x}}_0,\boldsymbol{\epsilon}}\left[\|\cdot\|_2^2\right]\right]
$$

is attained when the inner expectation

$$
\mathbb{E}_{{\mathbf{x}}_0,\boldsymbol{\epsilon}}\left[\|\cdot\|_2^2\right]
$$

is minimized for each fixed $t$. Since this is a standard mean squared error problem, the minimizer is unique. From denoising score matching ([Vincent, 2011](references.md#ref-vincent2011connection)), [Theorem 4.3.1](../../chapters/04-score-sde-framework/03-training-and-sampling.md#dsm-minimizer) shows the optimal score function satisfies

$$
\begin{aligned}
{\mathbf{s}}^*({\mathbf{x}}_t,t) = \mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_t)}\left[\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}}_t|{\mathbf{x}}_0)\right] = \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t).
\end{aligned}
$$

Using the identity $\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}}_t|{\mathbf{x}}_0) = -\frac{1}{\sigma_t} \boldsymbol{\epsilon}$ for $\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0},{\mathbf{I}})$, we obtain

$$
\begin{aligned}
{\mathbf{s}}^*({\mathbf{x}}_t,t) = -\frac{1}{\sigma_t}\, \boldsymbol{\epsilon}^*({\mathbf{x}}_t,t),
\end{aligned}
$$

where $\boldsymbol{\epsilon}^*({\mathbf{x}}_t,t) = \mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_t)}[\boldsymbol{\epsilon}|{\mathbf{x}}_t]$ is the optimal ${\boldsymbol{\epsilon}}$-prediction for $\mathcal{L}_{\text{noise}}({\boldsymbol{\phi}})$. For the ${\mathbf{x}}$-prediction loss $\mathcal{L}_{\text{clean}}$, the optimal estimator is

$$
{\mathbf{x}}^*({\mathbf{x}}_t,t) = \mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_t)}[{\mathbf{x}}_0|{\mathbf{x}}_t],
$$

which, by Tweedie’s formula, relates to the score via

$$
\alpha_t\, {\mathbf{x}}^*({\mathbf{x}}_t,t) = {\mathbf{x}}_t + \sigma_t^2\, {\mathbf{s}}^*({\mathbf{x}}_t,t).
$$

For the ${\mathbf{v}}$-prediction loss $\mathcal{L}_{\text{velocity}}$, the optimal estimator is

$$
\begin{aligned}
{\mathbf{v}}^*({\mathbf{x}}_t,t) = \mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_t)}[\alpha_t' {\mathbf{x}}_0 + \sigma_t' \boldsymbol{\epsilon}|{\mathbf{x}}_t]
    = \alpha_t' {\mathbf{x}}^* + \sigma_t' \boldsymbol{\epsilon}^*.
\end{aligned}
$$

Substituting the expressions for ${\mathbf{x}}^*$ and $\boldsymbol{\epsilon}^*$ in terms of ${\mathbf{s}}^*$ yields

$$
\begin{aligned}
{\mathbf{v}}^*({\mathbf{x}}_t,t)
    = \frac{\alpha_t'}{\alpha_t} {\mathbf{x}}_t + \left( \frac{\alpha_t'}{\alpha_t} \sigma_t^2 - \sigma_t \sigma_t' \right) {\mathbf{s}}^*({\mathbf{x}}_t,t) = f(t) {\mathbf{x}}_t - \frac{1}{2}g^2(t) {\mathbf{s}}^*({\mathbf{x}}_t,t),
\end{aligned}
$$

which completes the derivation.

$\blacksquare$
