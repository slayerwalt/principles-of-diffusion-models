# 9.7 Closing Remarks {#sec-ch9-cr}

This chapter has confronted one of the most significant practical limitations of diffusion models: their slow, iterative sampling process. We have explored training-free approaches that accelerate generation by designing better numerical solvers for the empirical PF-ODE:

1.  We began with DDIM, a first-order factored exponential-integrator update that retains the known diffusion schedule analytically while holding the neural prediction fixed.

2.  We then studied DEIS, which extends this idea to higher-order multistep updates by reusing past model evaluations to extrapolate how the neural prediction changes.

3.  Finally, we examined the DPM-Solver family, which introduces the log-SNR time coordinate and constructs higher-order updates from within-step model evaluations. DPM-Solver`++` further develops single-step and multistep variants in the data-prediction formulation.

Together, these methods illustrate a common principle: exploit the analytically known diffusion structure and concentrate numerical approximation on the learned model prediction. They differ in the prediction and time parameterizations, and in whether higher-order information is obtained from within-step stages or previous-step history. ParaDiGMS provides a complementary direction by reorganizing computation across time through fixed-point iteration and parallel evaluation. Such solver developments can reduce the sampling cost from hundreds or thousands of model evaluations to only tens in many practical settings.

However, these training-free methods remain fundamentally iterative: they numerically approximate a continuous generative trajectory, even when parts of the computation can be parallelized. This raises a natural question: *can we achieve high-quality generation in just one or a very few steps?*

The final Part D of this monograph explores this question through training-based acceleration. In [Chapter 10](../10-distillation/index.md#ch-distillation), we study distillation-based methods, where a fast *student* generator learns to reproduce the behavior of a slow pre-trained *teacher*. In [Chapter 11](../11-fast-generators/index.md#ch-fast-scratch), we turn to methods that directly learn few-step generators from scratch defined by an ODE flow.

This shift from designing a better numerical solver to directly learning a fast approximation of the generative solution map aims to combine the quality of diffusion models with the speed of one- or few-step generation.
