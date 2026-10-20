# 3. Score-Based Perspective: From EBMs to NCSN {#ch-score-based}

In the previous chapters we traced diffusion models to their variational roots and showed how they arise within the framework of VAEs. We now turn to a second, equally fundamental viewpoint: *Energy-Based Models* (EBMs) ([Ackley et al., 1985](../../references.md#ref-ackley1985learning); [LeCun et al., 2006](../../references.md#ref-lecun2006tutorial)). An EBM represents a distribution by an energy landscape that is low on data and high elsewhere. Sampling typically relies on Langevin dynamics, which moves samples toward high density regions by following the gradient of this landscape. This gradient field, known as the *score*, points toward directions of higher probability.

The central observation is that knowing the score is enough for generation: it moves samples toward likely regions without computing the intractable normalization constant. *Score-based* diffusion models build directly on this idea. Instead of focusing only on the clean data distribution, they consider a sequence of Gaussian noise–perturbed distributions whose scores are easier to approximate. Learning these scores yields a family of vector fields that guide noisy samples step by step back to data, turning generation into progressive denoising.

**After this chapter, you will be able to...**

-   understand the core idea of score-based modeling: what a score function is, and how it can be used with Langevin dynamics for generation;

-   understand how denoising score matching learns the score at a single noise level through conditioning trick, and how Tweedie’s formula connects score prediction to denoising;

-   understand how Noise Conditional Score Networks (NCSNs) extend denoising score matching to multiple noise levels, why multiple noise levels are useful, and how NCSN score prediction relates to DDPM noise prediction.

**Reader route.** Readers already familiar with energy-based models may skim [Section 3.1](01-energy-based-models.md#sec-ebm) and begin with [Section 3.2.1](02-score-based-models.md#subsec-training-sm).

Continue with the sections below and the [references](../../references.md#references).

<span id="sec-ebm"></span>

[3.1 Energy-Based Models](01-energy-based-models.md#sec-ebm)

<span id="eq-ebm-sm"></span>

[Equation (3.1.3)](01-energy-based-models.md#eq-ebm-sm)

<span id="sec-ebm-to-score"></span>

[3.2 From Energy-Based to Score-Based Generative Models](02-score-based-models.md#sec-ebm-to-score)

<span id="eq-sm"></span>

[Equation (3.2.1)](02-score-based-models.md#eq-sm)

<span id="sec-dsm"></span>

[3.3 Denoising Score Matching](03-denoising-score-matching.md#sec-dsm)

<span id="sec-smld"></span>

[3.4 Multi-Noise Levels of Denoising Score Matching (NCSN)](04-noise-conditional-score-networks.md#sec-smld)

<span id="sec-comparison-ncsn-ddpm"></span>

[3.5 Summary: A Comparative View of NCSN and DDPM](05-comparison.md#sec-comparison-ncsn-ddpm)

<span id="sec-ch3-cr"></span>

[3.6 Closing Remarks](06-closing-remarks.md#sec-ch3-cr)
