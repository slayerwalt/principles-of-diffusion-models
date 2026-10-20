# 8. Guidance and Controllable Generation {#ch-guidance}

Diffusion models are powerful generative frameworks. In the unconditional setting, the goal is to learn $p_{\text{data}}({\mathbf{x}})$ and generate samples without external input.

Many applications, however, require *conditional generation*, where outputs satisfy user-specified criteria. This can be achieved by steering an unconditional model or directly learning the conditional distribution $p_0({\mathbf{x}} | {\mathbf{c}})$, with condition ${\mathbf{c}}$ (e.g., label, text description, or sketch) guiding the process.

This chapter builds on a principled view of the conditional score, which decomposes into an *unconditional direction* and a *guidance direction* that nudges samples toward the condition while preserving realism. We explain why guidance is essential, show how the conditional score serves as a unifying interface for control, and survey ways to approximate or replace the guidance term. We then distinguish *control* (meeting the condition) from *alignment* (meeting human preference under the condition), and describe how preferences can be incorporated into the same framework. Finally, we discuss direct optimization of preference without additional reward models (i.e., a learned scorer that assigns higher values to outputs better aligned with human preference).

**After this chapter, you will be able to...**

-   understand how conditioning modifies the velocity field used during diffusion sampling;

-   derive the key ideas behind classifier guidance and classifier-free guidance, and understand how guidance strength controls conditioning;

-   understand how training-free guidance uses task-specific losses or measurement models to steer a pre-trained diffusion model, including for inverse problems;

-   distinguish exact conditional sampling from surrogate-guided dynamics and their fixed-time tilted-density interpretations;

-   understand how preference-based alignment, from RLHF to DPO and Diffusion-DPO, further reshapes a conditional generative model toward preferred outputs.

**Reader route.** The chapter assumes the Score SDE/PF-ODE picture from [Chapter 4](../04-score-sde-framework/index.md#ch-score-sde) and [Chapter 6](../06-unified-perspective/index.md#ch-unified-perspective). Readers interested primarily in controllable generation may treat the later preference-alignment material as a second pass.

Continue with the sections below and the [references](../../references.md#references).

[8.1 Prologue](01-prologue.md#sec-prologue-guidance)

[8.2 Classifier Guidance](02-classifier-guidance.md#sec-cg)

[8.3 Classifier-Free Guidance](03-classifier-free-guidance.md#sec-cfg)

[8.4 Training-Free Guidance](04-training-free-guidance.md#sec-training-free-guidance)

[8.5 From Reinforcement Learning to Direct Preference Optimization for Model Alignment](05-preference-alignment.md#sec-rlhf-dpo)

[8.6 Closing Remarks](06-closing-remarks.md#sec-ch8-cr)
