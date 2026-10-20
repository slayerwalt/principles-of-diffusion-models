# 9.5 DPM-Solver`++` {#sec-dpm}

## 9.5.1 From DPM-Solver to DPM-Solver`++` for Guided Sampling {#guided-sampling}

DPM-Solver`++` ([Lu et al., 2022](../../references.md#ref-lu2022dpm2)) was developed to improve high-order sampling under strong guidance. In this regime, two practical difficulties arise. First, large guidance scales can amplify the model output and its derivatives, narrowing the convergence region of high-order solvers and requiring smaller numerical steps. Second, the guided prediction may drive the final solution outside the range represented by the training data, leading to the train–test mismatch discussed in [Chapter 8](../08-guidance/index.md#ch-guidance).

DPM-Solver`++` addresses these issues in two complementary ways. It reformulates the diffusion ODE using data prediction, for which data-space corrections such as dynamic thresholding can be applied, and it develops a multistep variant that reuses previous model evaluations to obtain smaller effective step sizes under a fixed NFE budget. We first derive the data-prediction formulation and then introduce its single-step and multistep second-order solvers.

## 9.5.2 DPM-Solver`++`’s Methodology {#methodology}

DPM-Solver`++` keeps the log-SNR coordinate $\lambda$ introduced for DPM-Solver, but expresses the empirical PF-ODE through data prediction rather than noise prediction. Recall from [Equation (6.3.1)](../06-unified-perspective/03-equivalent-parameterizations.md#eq-predictions-equivalence) that

$$
{\boldsymbol{\epsilon}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t,t)
=
\frac{
{\mathbf{x}}_t-\alpha_t{\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t,t)
}{
\sigma_t
}.
$$

Substituting this relation into the noise-prediction exact solution in [Equation (9.4.5)](04-dpm-solver.md#eq-analytic-solution) gives the equivalent data-prediction form

<span id="eq-dpmpp-data-exact"></span>

$$
\widetilde{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
=
\frac{\sigma_t}{\sigma_s}{\mathbf{x}}_s
+
\sigma_t
\int_{\lambda_s}^{\lambda_t}
e^\lambda
\hat{{\mathbf{x}}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda)
\mathop{}\!\mathrm{d}\lambda,
\tag{9.5.1}
$$

where, following [Equation (9.4.3)](04-dpm-solver.md#eq-change-of-var-x-epsilon), we define the $\lambda$-parameterized data-prediction field by

$$
\hat{{\mathbf{x}}}_{\boldsymbol{\phi}^\times}
({\mathbf{x}},\lambda)
:=
{\mathbf{x}}_{\boldsymbol{\phi}^\times}
\bigl(
{\mathbf{x}},
t_\lambda(\lambda)
\bigr).
$$

Hence, along the empirical PF-ODE trajectory,

$$
\hat{{\mathbf{x}}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda)
=
{\mathbf{x}}_{\boldsymbol{\phi}^\times}
\bigl(
{\mathbf{x}}_{t_\lambda(\lambda)},
t_\lambda(\lambda)
\bigr).
$$

Thus, the structure is exactly dual to DPM-Solver: the known schedule-dependent evolution is handled analytically, while the only model-dependent quantity inside the remaining integral is now the data prediction. DPM-Solver`++` develops two second-order realizations of this exact form:

-   **Single-Step Solver, DPM-Solver`++`(2S).** Use an additional model evaluation at an intermediate stage within the current step, analogous to the staged construction of DPM-Solver-2.

-   **Multistep Solver, DPM-Solver`++`(2M).** Reuse data predictions from the previous two completed steps. After the warm start, only one new model evaluation is required per step.

**Remark.**

Without thresholding, converting a model output to data prediction is a pointwise algebraic re-expression of the same empirical PF-ODE. The numerical method can nevertheless change because the finite-step approximation is now applied to the data prediction rather than to the noise prediction. If thresholding is used, it is an additional modification of the model output and should be viewed separately from the numerical-order analysis below.

## 9.5.3 DPM-Solver`++` Single-Step by Taylor Expansion {#subsec-dpm-solver--single}

Following the DPM-Solver construction in [Section 9.4.2](04-dpm-solver.md#subsec-dpm-method-2), DPM-Solver`++` approximates the data prediction along the empirical trajectory by a local Taylor expansion in log-SNR time. For $k\geq0$, denote its $k$-th total derivative along the empirical PF-ODE trajectory by

$$
\hat{{\mathbf{x}}}_{\boldsymbol{\phi}^\times}^{(k)}
(\hat{\mathbf{x}}_\lambda,\lambda)
:=
\frac{\mathop{}\!\mathrm{d}^k}{\mathop{}\!\mathrm{d}\lambda^k}
\hat{{\mathbf{x}}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda).
$$

For one step from $t_{i-1}$ to $t_i$, let $h_i := \lambda_{t_i}-\lambda_{t_{i-1}} >0$. A Taylor expansion about $\lambda_{t_{i-1}}$ gives

$$
\hat{{\mathbf{x}}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda)
=
\sum_{k=0}^{n-1}
\frac{
(\lambda-\lambda_{t_{i-1}})^k
}{
k!
}
\hat{{\mathbf{x}}}_{\boldsymbol{\phi}^\times}^{(k)}
\bigl(
\hat{\mathbf{x}}_{\lambda_{t_{i-1}}},
\lambda_{t_{i-1}}
\bigr)
+
\mathcal O
\bigl(
(\lambda-\lambda_{t_{i-1}})^n
\bigr).
$$

Substituting this expansion into [Equation (9.5.1)](05-dpm-solver-plus-plus.md#eq-dpmpp-data-exact) yields

<span id="eq-dpmpp-taylor-expansion"></span>

$$
\begin{aligned}
\widetilde{\boldsymbol{\Psi}}_{t_{i-1}\to t_i}
(\tilde{\mathbf{x}}_{t_{i-1}})
=
\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}
\tilde{\mathbf{x}}_{t_{i-1}}
+
\alpha_{t_i}
\sum_{k=0}^{n-1}
h_i^{k+1}
\varphi_{k+1}(-h_i)
\hat{{\mathbf{x}}}_{\boldsymbol{\phi}^\times}^{(k)}
\bigl(
\hat{\mathbf{x}}_{\lambda_{t_{i-1}}},
\lambda_{t_{i-1}}
\bigr)
+
\mathcal O(h_i^{n+1}),
\end{aligned}
\tag{9.5.2}
$$

where $\varphi_m$ is defined in [Equation (9.4.9)](04-dpm-solver.md#eq-analytic-expansion). Here we used

$$
\int_{\lambda_{t_{i-1}}}^{\lambda_{t_i}}
e^\lambda
\frac{
(\lambda-\lambda_{t_{i-1}})^k
}{
k!
}
\mathop{}\!\mathrm{d}\lambda
=
e^{\lambda_{t_i}}
h_i^{k+1}
\varphi_{k+1}(-h_i),
$$

together with $\sigma_{t_i}e^{\lambda_{t_i}}=\alpha_{t_i}$.

For $n=1$, dropping the $\mathcal O(h_i^2)$ remainder gives

$$
\tilde{\mathbf{x}}_{t_i}
=
\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}
\tilde{\mathbf{x}}_{t_{i-1}}
+
\alpha_{t_i}
\left(
1-e^{-h_i}
\right)
{\mathbf{D}}_{i-1},
\qquad
{\mathbf{D}}_{i-1}
:=
{\mathbf{x}}_{\boldsymbol{\phi}^\times}
(\tilde{\mathbf{x}}_{t_{i-1}},t_{i-1}),
$$

which is exactly the deterministic DDIM update in data-prediction form.

For $n=2$, the first total derivative is approximated using an additional within-step stage. For the midpoint choice, define

$$
s_i^{\mathrm{mid}}
:=
t_\lambda
\left(
\lambda_{t_{i-1}}+\frac{h_i}{2}
\right).
$$

A first-order DPM-Solver`++` step predicts the midpoint state

$$
{\mathbf{u}}_i
=
\frac{
\sigma_{s_i^{\mathrm{mid}}}
}{
\sigma_{t_{i-1}}
}
\tilde{\mathbf{x}}_{t_{i-1}}
+
\alpha_{s_i^{\mathrm{mid}}}
\left(
1-e^{-h_i/2}
\right)
{\mathbf{D}}_{i-1},
$$

at which we evaluate

$$
{\mathbf{D}}_i^{\mathrm{mid}}
:=
{\mathbf{x}}_{\boldsymbol{\phi}^\times}
({\mathbf{u}}_i,s_i^{\mathrm{mid}}).
$$

The midpoint DPM-Solver`++`(2S) update then takes the particularly simple form

$$
\tilde{\mathbf{x}}_{t_i}
=
\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}
\tilde{\mathbf{x}}_{t_{i-1}}
+
\alpha_{t_i}
\left(
1-e^{-h_i}
\right)
{\mathbf{D}}_i^{\mathrm{mid}}.
$$

As with midpoint DPM-Solver-2, this simplified two-stage update differs from the direct second-order Taylor/finite-difference construction only by $\mathcal O(h_i^3)$ and therefore retains second-order accuracy.

This yields the single-step DPM-Solver`++` construction. When $n=1$, it reduces exactly to the deterministic DDIM update in data-prediction form. When $n=2$, approximating the first total $\lambda$-derivative using an additional within-step model evaluation gives *DPM-Solver`++`(2S)*, the data-prediction counterpart of DPM-Solver-2 in [Algorithm 5](04-dpm-solver.md#alg-dpm-solver-2). Its midpoint implementation is summarized in [Algorithm 6](05-dpm-solver-plus-plus.md#alg-dpmpp-2s-mid).

<section class="book-algorithm" id="alg-dpmpp-2s-mid" aria-labelledby="alg-dpmpp-2s-mid-title" markdown="1">

<div class="book-algorithm__title" id="alg-dpmpp-2s-mid-title" markdown="span">
Algorithm 6: DPM-Solver++(2S): a midpoint special case.
</div>

<div class="book-algorithm__meta" markdown="1">

**Input:** initial value ${\mathbf{x}}_T$, time steps $\{t_i\}_{i=0}^M$, data-prediction model ${\mathbf{x}}_{{\boldsymbol{\phi}}^\times}$

</div>

<div class="book-algorithm__scroll" tabindex="0" role="region" aria-label="Algorithm 6 steps" markdown="1">
<div class="book-algorithm__lines" markdown="1">

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">1</div>
<div class="book-algorithm__statement" markdown="1">

$\tilde{\mathbf{x}}_{t_0}\gets{\mathbf{x}}_T$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">2</div>
<div class="book-algorithm__statement" markdown="1">

$\lambda_{t_i}\gets\log(\alpha_{t_i}/\sigma_{t_i})$ for $i=0,\ldots,M$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">3</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{D}}_0 \gets {\mathbf{x}}_{{\boldsymbol{\phi}}^\times}(\tilde{\mathbf{x}}_{t_0},t_0)$  <span class="book-algorithm__comment">▷ *cache start prediction*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">4</div>
<div class="book-algorithm__statement" markdown="1">

**for** $i\gets1$ **to** $M$ **do**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">5</div>
<div class="book-algorithm__statement" markdown="1">

$h_i \gets \lambda_{t_i}-\lambda_{t_{i-1}}$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">6</div>
<div class="book-algorithm__statement" markdown="1">

$s_i^{\mathrm{mid}} \gets t_\lambda \left( \tfrac{\lambda_{t_{i-1}}+\lambda_{t_i}}{2} \right)$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">7</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{u}}_i \gets \tfrac{\sigma_{s_i^{\mathrm{mid}}}}{\sigma_{t_{i-1}}} \tilde{\mathbf{x}}_{t_{i-1}} + \alpha_{s_i^{\mathrm{mid}}} \left( 1-e^{-h_i/2} \right) {\mathbf{D}}_{i-1}$  <span class="book-algorithm__comment">▷ *predict midpoint stage*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">8</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{D}}_i^{\mathrm{mid}} \gets {\mathbf{x}}_{{\boldsymbol{\phi}}^\times} ({\mathbf{u}}_i,s_i^{\mathrm{mid}})$  <span class="book-algorithm__comment">▷ *evaluate at midpoint*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">9</div>
<div class="book-algorithm__statement" markdown="1">

$\tilde{\mathbf{x}}_{t_i} \gets \tfrac{\sigma_{t_i}}{\sigma_{t_{i-1}}} \tilde{\mathbf{x}}_{t_{i-1}} + \alpha_{t_i} \left( 1-e^{-h_i} \right) {\mathbf{D}}_i^{\mathrm{mid}}$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">10</div>
<div class="book-algorithm__statement" markdown="1">

**if** $i&lt;M$ **then**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 2" markdown="1">
<div class="book-algorithm__number">11</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{D}}_i \gets {\mathbf{x}}_{{\boldsymbol{\phi}}^\times} (\tilde{\mathbf{x}}_{t_i},t_i)$  <span class="book-algorithm__comment">▷ *cache for next step*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">12</div>
<div class="book-algorithm__statement" markdown="1">

**end if**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">13</div>
<div class="book-algorithm__statement" markdown="1">

**end for**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">14</div>
<div class="book-algorithm__statement" markdown="1">

**return** $\tilde{\mathbf{x}}_{t_M}$

</div>
</div>

</div>
</div>

</section>

## 9.5.4 DPM-Solver`++` Multistep by Recycling History {#subsec-dpm-solver--multi}

DPM-Solver`++`(2S) obtains second-order information by introducing an additional model evaluation *within the current step*. DPM-Solver`++`(2M) takes a different route: it reuses model predictions from previous completed steps to estimate the same local variation. After initialization, this requires only one new model evaluation per step, allowing more, and hence smaller, numerical steps under the same NFE budget. This is particularly useful under strong guidance, where large steps can reduce numerical stability.

The idea is simple. Recall that the second-order expansion in [Equation (9.5.2)](05-dpm-solver-plus-plus.md#eq-dpmpp-taylor-expansion) requires both the current data prediction and its first total derivative with respect to $\lambda$. The single-step method estimates this derivative using a new within-step stage; the multistep method instead estimates it from the recent history of model predictions.

For the first step, no such history is available, so we initialize with the first-order data-prediction update

$$
\tilde{\mathbf{x}}_{t_1}
=
\frac{\sigma_{t_1}}{\sigma_{t_0}}
\tilde{\mathbf{x}}_{t_0}
+
\alpha_{t_1}
\left(
1-e^{-h_1}
\right)
{\mathbf{D}}_0,
\qquad
{\mathbf{D}}_0
:=
{\mathbf{x}}_{\boldsymbol{\phi}^\times}
(\tilde{\mathbf{x}}_{t_0},t_0).
$$

This is exactly the deterministic DDIM update in data-prediction form.

Once two completed-step predictions are available, for $i\geq2$ define

$$
{\mathbf{D}}_{i-1}
:=
{\mathbf{x}}_{\boldsymbol{\phi}^\times}
(\tilde{\mathbf{x}}_{t_{i-1}},t_{i-1}),
\qquad
{\mathbf{D}}_{i-2}
:=
{\mathbf{x}}_{\boldsymbol{\phi}^\times}
(\tilde{\mathbf{x}}_{t_{i-2}},t_{i-2}).
$$

Keeping the first two terms of [Equation (9.5.2)](05-dpm-solver-plus-plus.md#eq-dpmpp-taylor-expansion) gives

$$
\begin{aligned}
\widetilde{\boldsymbol{\Psi}}_{t_{i-1}\to t_i}
(\tilde{\mathbf{x}}_{t_{i-1}})
=
\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}
\tilde{\mathbf{x}}_{t_{i-1}}
&+
\alpha_{t_i}
\left(
1-e^{-h_i}
\right)
{\mathbf{D}}_{i-1}
\\
&+
\alpha_{t_i}
\left(
h_i-1+e^{-h_i}
\right)
\hat{{\mathbf{x}}}_{\boldsymbol{\phi}^\times}^{(1)}
\bigl(
\hat{\mathbf{x}}_{\lambda_{t_{i-1}}},
\lambda_{t_{i-1}}
\bigr)
+
\mathcal O(h_i^3).
\end{aligned}
$$

Thus, the only additional quantity needed for a second-order update is the first total $\lambda$-derivative of the data prediction.

Instead of evaluating the model at a new intermediate stage, DPM-Solver`++`(2M) estimates this derivative from the previous completed step:

$$
\hat{{\mathbf{x}}}_{\boldsymbol{\phi}^\times}^{(1)}
\bigl(
\hat{\mathbf{x}}_{\lambda_{t_{i-1}}},
\lambda_{t_{i-1}}
\bigr)
\approx
\frac{
{\mathbf{D}}_{i-1}-{\mathbf{D}}_{i-2}
}{
h_{i-1}
}.
$$

Under standard smoothness assumptions, this backward difference has error $\mathcal O(h_{i-1})$. Substituting it into the second-order expansion and collecting the two prediction terms gives

$$
\tilde{\mathbf{x}}_{t_i}^{\mathrm{poly}}
=
\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}
\tilde{\mathbf{x}}_{t_{i-1}}
+
\alpha_{t_i}
\left(
1-e^{-h_i}
\right)
{\mathbf{D}}_i^{\mathrm{poly}},
$$

where

$$
{\mathbf{D}}_i^{\mathrm{poly}}
:=
{\mathbf{D}}_{i-1}
+
\frac{\beta(h_i)}{h_{i-1}}
\left(
{\mathbf{D}}_{i-1}-{\mathbf{D}}_{i-2}
\right),
\qquad
\beta(h)
:=
\frac{
h-1+e^{-h}
}{
1-e^{-h}
}.
$$

When consecutive step ratios remain bounded, this weighted multistep update has one-step local defect $\mathcal O(h_i^3)$.

**The Practical DPM-Solver`++`(2M) Update.** The expression above is already a valid second-order multistep construction. DPM-Solver`++`(2M) uses a slightly simpler coefficient that preserves the same order. Since

$$
\beta(h)
=
\frac{h}{2}
+
\frac{h^2}{12}
+
\mathcal O(h^3),
$$

define the step ratio

$$
r_i
:=
\frac{h_i}{h_{i-1}}
$$

and replace ${\mathbf{D}}_i^{\mathrm{poly}}$ by

$$
{\mathbf{D}}_i^{\mathrm{sim}}
:=
{\mathbf{D}}_{i-1}
+
\frac{r_i}{2}
\left(
{\mathbf{D}}_{i-1}-{\mathbf{D}}_{i-2}
\right).
$$

Equivalently,

$$
{\mathbf{D}}_i^{\mathrm{sim}}
=
\left(
1+\frac{r_i}{2}
\right){\mathbf{D}}_{i-1}
-
\frac{r_i}{2}{\mathbf{D}}_{i-2}.
$$

The interpretation is straightforward: the two most recent model predictions tell us the local direction in which the data prediction is changing, and ${\mathbf{D}}_i^{\mathrm{sim}}$ extrapolates that trend into the current step.

Under the same smoothness and bounded-step-ratio assumptions,

$$
{\mathbf{D}}_i^{\mathrm{poly}}
-
{\mathbf{D}}_i^{\mathrm{sim}}
=
\mathcal O(h_i^2).
$$

Since

$$
\alpha_{t_i}
\left(
1-e^{-h_i}
\right)
=
\mathcal O(h_i),
$$

this simplification changes the state update only by $\mathcal O(h_i^3)$. Hence, it preserves second-order accuracy, giving the practical DPM-Solver`++`(2M) update

$$
\tilde{\mathbf{x}}_{t_i}
=
\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}
\tilde{\mathbf{x}}_{t_{i-1}}
+
\alpha_{t_i}
\left(
1-e^{-h_i}
\right)
{\mathbf{D}}_i^{\mathrm{sim}}.
$$

After computing $\tilde{\mathbf{x}}_{t_i}$, the model is evaluated once at the new state,

$$
{\mathbf{D}}_i
=
{\mathbf{x}}_{\boldsymbol{\phi}^\times}
(\tilde{\mathbf{x}}_{t_i},t_i),
$$

and this prediction is cached for the next step. Thus, after the warm start, each DPM-Solver`++`(2M) step requires only one new model evaluation.

**Remark.**

For uniform log-SNR steps, $r_i=1$, so

$$
{\mathbf{D}}_i^{\mathrm{sim}}
=
\frac{3}{2}{\mathbf{D}}_{i-1}
-
\frac{1}{2}{\mathbf{D}}_{i-2}.
$$

The coefficients $(\frac32,-\frac12)$ are the familiar AB2 extrapolation weights for the *model prediction*. The complete DPM-Solver`++`(2M) update is nevertheless not plain AB2: the known diffusion-dependent factor $\alpha_{t_i}(1-e^{-h_i})$ is retained analytically.

The complete warm-start and two-step procedure is summarized in [Algorithm 7](05-dpm-solver-plus-plus.md#alg-dpmpp-2m). The key distinction from DPM-Solver`++`(2S) is where the second-order information comes from: 2S obtains it from an additional stage within the current step, whereas 2M recycles model evaluations from previous completed steps.

<section class="book-algorithm" id="alg-dpmpp-2m" aria-labelledby="alg-dpmpp-2m-title" markdown="1">

<div class="book-algorithm__title" id="alg-dpmpp-2m-title" markdown="span">
Algorithm 7: DPM-Solver++(2M).
</div>

<div class="book-algorithm__meta" markdown="1">

**Input:** initial value ${\mathbf{x}}_T$, time steps $\{t_i\}_{i=0}^M$, data-prediction model ${\mathbf{x}}_{{\boldsymbol{\phi}}^\times}$

</div>

<div class="book-algorithm__scroll" tabindex="0" role="region" aria-label="Algorithm 7 steps" markdown="1">
<div class="book-algorithm__lines" markdown="1">

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">1</div>
<div class="book-algorithm__statement" markdown="1">

$\tilde{\mathbf{x}}_{t_0}\gets{\mathbf{x}}_T$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">2</div>
<div class="book-algorithm__statement" markdown="1">

$\lambda_{t_i}\gets\log(\alpha_{t_i}/\sigma_{t_i})$ for $i=0,\ldots,M$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">3</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{D}}_0 \gets {\mathbf{x}}_{{\boldsymbol{\phi}}^\times} (\tilde{\mathbf{x}}_{t_0},t_0)$  <span class="book-algorithm__comment">▷ *cache start prediction*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number"></div>
<div class="book-algorithm__statement" markdown="1">

**Case I. Warm start ($i=1$)**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">4</div>
<div class="book-algorithm__statement" markdown="1">

$h_1 \gets \lambda_{t_1}-\lambda_{t_0}$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">5</div>
<div class="book-algorithm__statement" markdown="1">

$\tilde{\mathbf{x}}_{t_1} \gets \tfrac{\sigma_{t_1}}{\sigma_{t_0}} \tilde{\mathbf{x}}_{t_0} + \alpha_{t_1} \left( 1-e^{-h_1} \right) {\mathbf{D}}_0$  <span class="book-algorithm__comment">▷ *DDIM in data prediction*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">6</div>
<div class="book-algorithm__statement" markdown="1">

**if** $M&gt;1$ **then**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">7</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{D}}_1 \gets {\mathbf{x}}_{{\boldsymbol{\phi}}^\times} (\tilde{\mathbf{x}}_{t_1},t_1)$  <span class="book-algorithm__comment">▷ *cache for multistep update*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">8</div>
<div class="book-algorithm__statement" markdown="1">

**end if**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number"></div>
<div class="book-algorithm__statement" markdown="1">

**Case II. Two-step multistep update ($i\geq2$)**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">9</div>
<div class="book-algorithm__statement" markdown="1">

**for** $i\gets2$ **to** $M$ **do**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">10</div>
<div class="book-algorithm__statement" markdown="1">

$h_i \gets \lambda_{t_i}-\lambda_{t_{i-1}}$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">11</div>
<div class="book-algorithm__statement" markdown="1">

$r_i \gets h_i/h_{i-1}$  <span class="book-algorithm__comment">▷ *step ratio*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">12</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{D}}_i^{\mathrm{sim}} \gets \left( 1+\tfrac12 r_i \right) {\mathbf{D}}_{i-1} - \tfrac12 r_i {\mathbf{D}}_{i-2}$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">13</div>
<div class="book-algorithm__statement" markdown="1">

$\tilde{\mathbf{x}}_{t_i} \gets \tfrac{\sigma_{t_i}}{\sigma_{t_{i-1}}} \tilde{\mathbf{x}}_{t_{i-1}} + \alpha_{t_i} \left( 1-e^{-h_i} \right) {\mathbf{D}}_i^{\mathrm{sim}}$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">14</div>
<div class="book-algorithm__statement" markdown="1">

**if** $i&lt;M$ **then**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 2" markdown="1">
<div class="book-algorithm__number">15</div>
<div class="book-algorithm__statement" markdown="1">

${\mathbf{D}}_i \gets {\mathbf{x}}_{{\boldsymbol{\phi}}^\times} (\tilde{\mathbf{x}}_{t_i},t_i)$  <span class="book-algorithm__comment">▷ *one new model call; cache for next step*</span>

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">16</div>
<div class="book-algorithm__statement" markdown="1">

**end if**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">17</div>
<div class="book-algorithm__statement" markdown="1">

**end for**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">18</div>
<div class="book-algorithm__statement" markdown="1">

**return** $\tilde{\mathbf{x}}_{t_M}$

</div>
</div>

</div>
</div>

</section>

## 9.5.5 Relation of DPM-Solver`++`(2M) to DEIS {#subsec-dpm-deis}

DPM-Solver`++`(2M) and DEIS use the same basic multistep idea: reuse model predictions from previous completed steps to approximate how the neural prediction changes over the next interval, while keeping the known diffusion-dependent weights analytic. The difference is the representation in which this idea is applied. DEIS was formulated earlier using noise prediction in the original diffusion time, whereas DPM-Solver`++`(2M) uses data prediction in log-SNR time.

To expose the connection directly, we can apply the DEIS polynomial extrapolation principle to the exact data-prediction integral underlying DPM-Solver`++`. Recall that

$$
\widetilde{\boldsymbol{\Psi}}_{t_{i-1}\to t_i}
(\tilde{\mathbf{x}}_{t_{i-1}})
=
\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}
\tilde{\mathbf{x}}_{t_{i-1}}
+
\sigma_{t_i}
\int_{\lambda_{t_{i-1}}}^{\lambda_{t_i}}
e^\lambda
\hat{{\mathbf{x}}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda)
\mathop{}\!\mathrm{d}\lambda.
$$

The known weight $e^\lambda$ is already handled analytically. The only quantity that must be approximated over the step is therefore the data prediction

$$
\lambda
\longmapsto
\hat{{\mathbf{x}}}_{\boldsymbol{\phi}^\times}
(\hat{\mathbf{x}}_\lambda,\lambda).
$$

Using the two most recent cached predictions ${\mathbf{D}}_{i-2}$ and ${\mathbf{D}}_{i-1}$, the simplest nonconstant approximation is the unique affine polynomial through these two anchors. In Lagrange form,

$$
P_1(\lambda)
=
\frac{
\lambda-\lambda_{t_{i-2}}
}{
h_{i-1}
}
{\mathbf{D}}_{i-1}
-
\frac{
\lambda-\lambda_{t_{i-1}}
}{
h_{i-1}
}
{\mathbf{D}}_{i-2}.
$$

The same polynomial can be written more transparently around the most recent anchor as

$$
P_1(\lambda)
=
{\mathbf{D}}_{i-1}
+
\frac{
\lambda-\lambda_{t_{i-1}}
}{
h_{i-1}
}
\left(
{\mathbf{D}}_{i-1}-{\mathbf{D}}_{i-2}
\right).
$$

The first expression is the Lagrange form, while the second is the backward-Newton form of the *same* affine polynomial. Thus, Lagrange interpolation and backward differences are not different numerical approximations here; they are simply two representations of the same two-anchor extrapolant.

Applying this extrapolant inside the DPM-Solver`++` integral gives

$$
\tilde{\mathbf{x}}_{t_i}^{\mathrm{poly}}
=
\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}
\tilde{\mathbf{x}}_{t_{i-1}}
+
\sigma_{t_i}
\int_{\lambda_{t_{i-1}}}^{\lambda_{t_i}}
e^\lambda
P_1(\lambda)
\mathop{}\!\mathrm{d}\lambda.
$$

Because the weight $e^\lambda$ is known, the two required moments can be evaluated exactly:

$$
\sigma_{t_i}
\int_{\lambda_{t_{i-1}}}^{\lambda_{t_i}}
e^\lambda\mathop{}\!\mathrm{d}\lambda
=
\alpha_{t_i}
\left(
1-e^{-h_i}
\right),
$$

and

$$
\sigma_{t_i}
\int_{\lambda_{t_{i-1}}}^{\lambda_{t_i}}
e^\lambda
\left(
\lambda-\lambda_{t_{i-1}}
\right)
\mathop{}\!\mathrm{d}\lambda
=
\alpha_{t_i}
\left(
h_i-1+e^{-h_i}
\right).
$$

Substituting these coefficients yields

$$
\begin{aligned}
\tilde{\mathbf{x}}_{t_i}^{\mathrm{poly}}
=
\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}
\tilde{\mathbf{x}}_{t_{i-1}}
&+
\alpha_{t_i}
\left(
1-e^{-h_i}
\right)
{\mathbf{D}}_{i-1}
\\
&+
\alpha_{t_i}
\left(
h_i-1+e^{-h_i}
\right)
\frac{
{\mathbf{D}}_{i-1}-{\mathbf{D}}_{i-2}
}{
h_{i-1}
}.
\end{aligned}
$$

Equivalently,

$$
\tilde{\mathbf{x}}_{t_i}^{\mathrm{poly}}
=
\frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}
\tilde{\mathbf{x}}_{t_{i-1}}
+
\alpha_{t_i}
\left(
1-e^{-h_i}
\right)
{\mathbf{D}}_i^{\mathrm{poly}},
$$

where ${\mathbf{D}}_i^{\mathrm{poly}}$ is defined in [Section 9.5.4](05-dpm-solver-plus-plus.md#subsec-dpm-solver--multi).

This shows the precise connection to DEIS. Before the practical coefficient simplification, the DPM-Solver`++` second-order multistep update is exactly what one obtains by applying the degree-one DEIS extrapolation principle to the data-prediction/log-SNR integral. In this sense, it is the data-prediction/log-SNR analogue of AB-DEIS-1.

The practical DPM-Solver`++`(2M) update further simplifies this coefficient using the leading-order expansion

$$
\beta(h_i)
=
\frac{h_i}{2}
+
\mathcal O(h_i^2),
$$

which gives

$$
\frac{\beta(h_i)}{h_{i-1}}
=
\frac{h_i}{2h_{i-1}}
+
\mathcal O\left(
\frac{h_i^2}{h_{i-1}}
\right).
$$

Accordingly, DPM-Solver`++`(2M) replaces $\beta(h_i)/h_{i-1}$ by $h_i/(2h_{i-1})$ in the effective prediction. Since

$$
{\mathbf{D}}_{i-1}-{\mathbf{D}}_{i-2}
=
\mathcal O(h_{i-1}),
$$

this changes the effective prediction by

$$
\mathcal O\left(
\frac{h_i^2}{h_{i-1}}
\right)
\mathcal O(h_{i-1})
=
\mathcal O(h_i^2).
$$

The coefficient multiplying the effective prediction in the state update is $\alpha_{t_i}(1-e^{-h_i})=\mathcal O(h_i)$, so the resulting change in the state update is only $\mathcal O(h_i^3)$. Therefore, DPM-Solver`++`(2M) and the degree-one weighted-polynomial update are generally different finite-step formulas, but they agree through second order.

This also highlights why the connection to DEIS is specific to DPM-Solver`++`(2M): 2M obtains its additional second-order information by extrapolating previous completed-step predictions, whereas DPM-Solver`++`(2S) obtains it from a new intermediate stage within the current step.

The complete DPM-Solver`++`(2M) procedure, including its first-order warm start and the caching of one new data prediction after each step, is summarized in [Algorithm 7](05-dpm-solver-plus-plus.md#alg-dpmpp-2m).

## 9.5.6 Closing Remark {#closing-remark}

Although our main discussion has focused on DPM-Solver and DPM-Solver`++`, it is useful to briefly place them in a broader context. DPM-Solver is derived in the noise-prediction formulation, whereas DPM-Solver`++` reformulates the solver using data prediction, which can behave more favorably under strong CFG. DPM-Solver-v3 ([Zheng et al., 2023](../../references.md#ref-zheng2023dpm)) takes this idea further by introducing a more general model representation whose coefficients are chosen using precomputed model statistics to reduce local discretization error. In this sense, it makes the representation itself part of the numerical solver design.

Another related direction is the predictor–corrector framework. In the Score SDE framework (see [Chapter 4](../04-score-sde-framework/index.md#ch-score-sde)), a *predictor* advances the sample using a numerical step of the reverse-time dynamics, while a *corrector*, typically a Langevin-type update at the same noise level, further refines the sample using the score (see [Section 3.4](../03-score-based-perspective/04-noise-conditional-score-networks.md#sec-smld)). This predict–then–refine pattern also has a classical numerical analogue. For example, Heun’s method first predicts an endpoint with Euler and then improves the numerical update using an averaged slope. Methods such as UniPC ([Zhao et al., 2023](../../references.md#ref-zhao2023unipc)) develop predictor–corrector constructions specifically for diffusion sampling. We do not pursue these extensions here, but they illustrate the same broader theme of this chapter: effective diffusion solvers combine classical numerical ideas with structure specific to the diffusion PF-ODE.
