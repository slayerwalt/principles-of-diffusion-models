# D.3 Flow-Based Perspective {#app-sec-flow-proof}

## D.3.1 [Lemma 5.1.1](../../chapters/05-flow-based-perspective/01-normalizing-flows-and-neural-odes.md#instant-change-of-var): Instantaneous Change of Variables {#instantaneous-change-of-variables}

***Proof.***

**Approach 1: Change-of-Variables Formula.** We denote $p({\mathbf{x}}(t), t)$ by $p_t({\mathbf{x}}_t)$. Starting from the ODE discretization

$$
{\mathbf{z}}_{t+\Delta t} = {\mathbf{z}}_t + \Delta t  {\mathbf{F}}({\mathbf{z}}_t, t),
$$

the change-of-variables formula for normalizing flows ([Equation (5.1.1)](../../chapters/05-flow-based-perspective/01-normalizing-flows-and-neural-odes.md#eq-nf-change-of-var-log)) gives

$$
\begin{aligned}
\log p_{t+\Delta t}({\mathbf{z}}_{t+\Delta t})
    &= \log p_t({\mathbf{z}}_t) - \log \Big| \det\big({\mathbf{I}} + \Delta t \nabla_{{\mathbf{z}}} {\mathbf{F}}({\mathbf{z}}_t, t)\big) \Big| \\
    &= \log p_t({\mathbf{z}}_t) - \operatorname{Tr}\Big(\log({\mathbf{I}} + \Delta t \nabla_{{\mathbf{z}}} {\mathbf{F}}({\mathbf{z}}_t, t)) \Big) \\
    &= \log p_t({\mathbf{z}}_t) - \Delta t  \operatorname{Tr}\big(\nabla_{{\mathbf{z}}} {\mathbf{F}}({\mathbf{z}}_t, t)\big) + \mathcal{O}(\Delta t^2),
\end{aligned}
$$

where we used $\log \det = \operatorname{Tr} \log$ and the expansion for small $\Delta t$. Taking the limit $\Delta t \to 0$ yields the continuous-time differential equation for the log-density. Indeed, the same trick is applied in [Equation (5.1.6)](../../chapters/05-flow-based-perspective/01-normalizing-flows-and-neural-odes.md#eq-log-det-tr).

**Approach 2: Continuity Equation.** We can also leverage the continuity equation, which essentially serves as the change-of-variables formula:

$$
\partial_t p({\mathbf{z}},t) = - \nabla_{{\mathbf{z}}} \cdot \big({\mathbf{F}}({\mathbf{z}},t) p({\mathbf{z}},t)\big).
$$

Expanding the divergence,

$$
\partial_t p = - \big( (\nabla_{{\mathbf{z}}} \cdot {\mathbf{F}}) p + {\mathbf{F}} \cdot \nabla_{{\mathbf{z}}} p \big).
$$

Along trajectories ${\mathbf{z}}(t)$ satisfying $\frac{\mathop{}\!\mathrm{d} {\mathbf{z}}}{\mathop{}\!\mathrm{d} t} = {\mathbf{F}}({\mathbf{z}}(t), t)$, the total time derivative is

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} p({\mathbf{z}}(t), t)
&= \nabla_{{\mathbf{z}}} p \cdot \frac{\mathop{}\!\mathrm{d} {\mathbf{z}}}{\mathop{}\!\mathrm{d} t} + \partial_t p \\
&= \nabla_{{\mathbf{z}}} p \cdot {\mathbf{F}} - \big( (\nabla_{{\mathbf{z}}} \cdot {\mathbf{F}}) p + {\mathbf{F}} \cdot \nabla_{{\mathbf{z}}} p \big) \\
&= - (\nabla_{{\mathbf{z}}} \cdot {\mathbf{F}}) p.
\end{aligned}
$$

Dividing by $p({\mathbf{z}}(t), t)$, we conclude

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} \log p({\mathbf{z}}(t), t) = - \nabla_{{\mathbf{z}}} \cdot {\mathbf{F}}({\mathbf{z}}(t), t).
$$

$\blacksquare$

## D.3.2 [Theorem 5.2.2](../../chapters/05-flow-based-perspective/02-flow-matching-framework.md#thm-continuity-mass): Mass Conservation Criterion {#app-sec-mass-conservation}

**Some Prerequisites: Flow Map and Flow-Induced Density.** For any initial position $\mathbf{x}_0\in\mathbb{R}^D$, the flow map $\boldsymbol{\Psi}_t:\mathbb{R}^D\to\mathbb{R}^D$ is the unique solution of the ODE

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} \boldsymbol{\Psi}_t(\mathbf{x}_0)
= \mathbf{v}_t\bigl(\boldsymbol{\Psi}_t(\mathbf{x}_0)\bigr),
\quad \boldsymbol{\Psi}_0(\mathbf{x}_0)=\mathbf{x}_0.
$$

Under our regularity assumptions, $\boldsymbol{\Psi}_t$ is continuously differentiable in both $t$ and $\mathbf{x}_0$.

The flow‑induced density $p_t^{\mathrm{fwd}}$ is the pushforward of the initial density $p_0$ by $\boldsymbol{\Psi}_t$:

$$
p_t^{\mathrm{fwd}}(\mathbf{x})
= p_0\bigl(\boldsymbol{\Psi}_t^{-1}(\mathbf{x})\bigr)
  \Bigl|\det \bigl(\nabla\boldsymbol{\Psi}_t^{-1}(\mathbf{x})\bigr)\Bigr|.
$$

This gives the density at $\mathbf{x}$ and time $t$ of particles that started from $p_0=p_{\mathrm{data}}$ and evolved under the velocity field $\mathbf{v}_t$.

***Informal Proof:*** **Sufficient Condition: $p_t^{\mathrm{fwd}} = p_t$ $\Rightarrow$ Continuity Equation.** In [Section B.1.2](../b-density-evolution/01-change-of-variable.md#subsec-cov-continuity-eq) we obtained a *strong solution* of the continuity equation by taking the continuous time limit of the discrete change of variable formula. In that approach, the density $p_t$ is assumed smooth enough that all derivatives exist classically and the PDE holds pointwise. Here, we offer a complementary derivation in the *weak sense*: the continuity equation is imposed only after integrating against arbitrary smooth test functions, which relaxes the regularity requirements on both $p_t$ and the velocity field $\mathbf{v}_t$. This weak formulation is not only more rigorous since it accommodates less regular solutions but is also the standard framework in PDE theory and numerical analysis.

For any smooth test function $\varphi(\mathbf{x})$ with compact support, the pushforward property gives:

$$
\begin{aligned}
\int p_t^{\mathrm{fwd}}(\mathbf{x}) \varphi(\mathbf{x})  \mathop{}\!\mathrm{d}\mathbf{x} &= \int p_0(\boldsymbol{\Psi}_t^{-1}(\mathbf{x})) \left| \det\left( \nabla \boldsymbol{\Psi}_t^{-1}(\mathbf{x}) \right) \right| \varphi(\mathbf{x}) \mathop{}\!\mathrm{d}\mathbf{x}\\ &=  \int p_0(\mathbf{y}) \varphi(\boldsymbol{\Psi}_t(\mathbf{y})) \mathop{}\!\mathrm{d}\mathbf{y},
\end{aligned}
$$

where the second equality follows from the change of variables $\mathbf{x} = \boldsymbol{\Psi}_t(\mathbf{y})$, with $\mathop{}\!\mathrm{d}\mathbf{y} = \left| \det\left( \nabla \boldsymbol{\Psi}_t^{-1}(\mathbf{x}) \right) \right| \mathop{}\!\mathrm{d}\mathbf{x}$.

Differentiate both sides with respect to $t$:

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} \int p_t^{\mathrm{fwd}}(\mathbf{x}) \varphi(\mathbf{x}) \mathop{}\!\mathrm{d}\mathbf{x} = \frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t} \int p_0(\mathbf{y}) \varphi(\boldsymbol{\Psi}_t(\mathbf{y})) \mathop{}\!\mathrm{d}\mathbf{y}.
$$

The left-hand side is:

$$
\int \frac{\partial p_t^{\mathrm{fwd}}}{\partial t}(\mathbf{x}) \varphi(\mathbf{x}) \mathop{}\!\mathrm{d}\mathbf{x}.
$$

On the right-hand side:

$$
\int p_0(\mathbf{y}) \nabla \varphi(\boldsymbol{\Psi}_t(\mathbf{y})) \cdot \mathbf{v}_t(\boldsymbol{\Psi}_t(\mathbf{y})) \mathop{}\!\mathrm{d}\mathbf{y},
$$

since $\frac{\partial \boldsymbol{\Psi}_t}{\partial t}(\mathbf{y}) = \mathbf{v}_t(\boldsymbol{\Psi}_t(\mathbf{y}))$. Change variables to $\mathbf{x} = \boldsymbol{\Psi}_t(\mathbf{y})$, so

$$
\mathop{}\!\mathrm{d}\mathbf{y} = \left| \det\left( \nabla \boldsymbol{\Psi}_t^{-1}(\mathbf{x}) \right) \right| \mathop{}\!\mathrm{d}\mathbf{x} ,
$$

and

$$
p_0(\mathbf{y}) = p_t^{\mathrm{fwd}}(\mathbf{x}) \left| \det\left( \nabla \boldsymbol{\Psi}_t(\mathbf{y}) \right) \right| = \frac{p_t^{\mathrm{fwd}}(\mathbf{x})}{\left| \det\left( \nabla \boldsymbol{\Psi}_t^{-1}(\mathbf{x}) \right) \right|}.
$$

Thus, the right-hand side becomes:

$$
\int p_t^{\mathrm{fwd}}(\mathbf{x}) \nabla \varphi(\mathbf{x}) \cdot \mathbf{v}_t(\mathbf{x}) \mathop{}\!\mathrm{d}\mathbf{x}.
$$

Apply integration by parts, using the compact support of $\varphi$:

$$
\int p_t^{\mathrm{fwd}}(\mathbf{x}) \nabla \varphi(\mathbf{x}) \cdot \mathbf{v}_t(\mathbf{x}) \mathop{}\!\mathrm{d}\mathbf{x} = -\int \varphi(\mathbf{x}) \nabla \cdot (p_t^{\mathrm{fwd}}(\mathbf{x}) \mathbf{v}_t(\mathbf{x})) \mathop{}\!\mathrm{d}\mathbf{x}.
$$

Equating both sides:

$$
\int \left[ \frac{\partial p_t^{\mathrm{fwd}}}{\partial t}(\mathbf{x}) + \nabla \cdot (p_t^{\mathrm{fwd}}(\mathbf{x}) \mathbf{v}_t(\mathbf{x})) \right] \varphi(\mathbf{x}) \mathop{}\!\mathrm{d}\mathbf{x} = 0.
$$

Since $\varphi$ is arbitrary, we conclude:

$$
\frac{\partial p_t^{\mathrm{fwd}}}{\partial t} + \nabla \cdot (p_t^{\mathrm{fwd}} \mathbf{v}_t) = 0.
$$

Given $p_t^{\mathrm{fwd}} = p_t$, this implies:

$$
\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t \mathbf{v}_t) = 0.
$$

**Necessary Condition: Continuity Equation $\Rightarrow p_t^{\mathrm{fwd}} = p_t$.** Suppose $p_t$ satisfies the continuity equation:

$$
\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t \mathbf{v}_t) = 0,
$$

with the initial condition $p_0(\mathbf{x}) = p_{\mathrm{data}}(\mathbf{x})$. We know $p_t^{\mathrm{fwd}}$ satisfies the same continuity equation, as shown above, and:

$$
p_0^{\mathrm{fwd}}(\mathbf{x}) = p_0(\boldsymbol{\Psi}_0^{-1}(\mathbf{x})) \left| \det\left( \nabla \boldsymbol{\Psi}_0^{-1}(\mathbf{x}) \right) \right| = p_0(\mathbf{x}),
$$

since $\boldsymbol{\Psi}_0(\mathbf{x}) = \mathbf{x}$. Thus, both densities share the same initial condition $p_0 = p_{\mathrm{data}}$.

The continuity equation can be rewritten as:

$$
\frac{\partial p}{\partial t} + \mathbf{v}_t \cdot \nabla p + p \nabla \cdot \mathbf{v}_t = 0.
$$

This is a first-order linear PDE. Assuming $\mathbf{v}_t$ is continuously differentiable and globally Lipschitz, and $p_t$ is sufficiently smooth, the method of characteristics guarantees a unique solution in the space of smooth functions. Since $p_t$ and $p_t^{\mathrm{fwd}}$ satisfy the same PDE and initial condition, we conclude:

$$
p_t(\mathbf{x}) = p_t^{\mathrm{fwd}}(\mathbf{x})
$$

for all $t\in[0,1]$ and $\mathbf{x} \in \mathbb{R}^D$.

This completes the proof of the equivalence. $\blacksquare$
