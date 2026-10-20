# 10. Distillation-Based Methods for Fast Sampling {#ch-distillation}

This chapter introduces training-based approaches that accelerate diffusion model sampling by teaching new generators to produce samples in only one or a few steps. The central idea, called *distillation*, is to let a fast student model learn from a slow, pre-trained diffusion model (teacher) sampler. While the teacher may require hundreds of steps, the student can achieve comparable quality in only a few steps[^1]. Unlike solver-based acceleration, which improves the numerical integration scheme, distillation directly trains a generator to take efficient shortcuts. We highlight two main paradigms: *distribution level distillation*, which skips simulating the full trajectory and instead aligns the student’s output distribution with the teacher’s, and *flow map level distillation*, which trains the student to reproduce the teacher’s sampling path in a faster and more compact way.

**After this chapter, you will be able to...**

-   distinguish the two main distillation targets: the teacher’s distribution and its state-to-state flow map;

-   understand distribution-level distillation, where the student matches the teacher’s noisy marginals through score-based supervision;

-   understand flow-map distillation, where the student learns teacher-generated transitions, with Progressive Distillation progressively replacing two teacher steps by one student step.

**Reader route.** With a pre-trained diffusion model as the teacher, the key distinction in this chapter is what the student is trained to reproduce: the teacher’s distribution or its flow map. The next chapter asks how such fast maps can be learned without a teacher.

[^1]: Here, distillation means reducing sampling steps rather than model size.

Continue with the sections below and the [references](../../references.md#references).

[10.1 Prologue](01-prologue.md#sec-distillation-prologue)

[10.2 Distribution-Based Distillation](02-distribution-based-distillation.md#sec-vsd)

[10.3 Progressive Distillation](03-progressive-distillation.md#sec-pd)

[10.4 Closing Remarks](04-closing-remarks.md#sec-ch10-cr)
