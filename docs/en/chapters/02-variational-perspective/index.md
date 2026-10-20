# 2. Variational Perspective: From VAEs to DDPMs {#ch-variational}

In this chapter we view diffusion models through a variational lens. We begin with the Variational Autoencoders (VAEs), which represent data with latent variables and are trained by maximizing a tractable lower bound on the log likelihood. In this setting a learned encoder maps observations to latents, and a learned decoder maps latents back to observations, closing the modeling loop.

Building on this pattern, hierarchical variants (Hierarchical VAEs) stack several latent layers to capture structure at multiple scales. With this setup, Denoising Diffusion Probabilistic Models (DDPM) conceptually follow the similar template: instead of jointly training both the encoder and decoder, the encoder is fixed as a forward noising process that gradually maps data to noise, and training learns a decoder that reverses this path in successive denoising steps. In this view, VAEs, hierarchical VAEs, and diffusion models all optimize a likelihood surrogate defined by a variational bound, providing a common foundation for the methods introduced here.

**After this chapter, you will be able to...**

-   understand why VAEs optimize the ELBO when exact maximum-likelihood training is difficult;

-   see how DDPM extends the latent-variable view of VAEs into many noisy layers, with a fixed noising process and a learned reverse process;

-   understand the conditioning trick that turns the difficult reverse-denoising problem into a tractable training target;

-   compare mean, clean-data, and noise prediction, and see how they represent the same underlying denoising information;

-   connect the practical DDPM denoising loss back to its variational foundation through the ELBO.

**Reader route.** Readers already familiar with VAEs and ELBO training may skim [Section 2.1](01-variational-autoencoder.md#sec-vae) and begin with [Section 2.2](02-ddpm.md#sec-ddpm).

<span id="sec-vae"></span>

Continue with [2.1 Variational Autoencoder](01-variational-autoencoder.md#sec-vae), [2.2 Variational Perspective: DDPM](02-ddpm.md#sec-ddpm), and [2.3 Closing Remarks](03-closing-remarks.md#sec-ch2-cr). See also the [references](../../references.md#references).
