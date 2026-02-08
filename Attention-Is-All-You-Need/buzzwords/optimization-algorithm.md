# Optimization Algorithm

An optimization algorithm updates model parameters to minimize a loss function. It determines how gradients translate into weight updates.

**In the Transformer**
The paper uses Adam with a custom learning-rate schedule that includes a warmup phase followed by a decay. This improves stability early in training.

**Why It Matters**
The optimizer and learning-rate schedule strongly affect convergence speed, stability, and final quality.

**Key Points**
1. Adaptive optimizers like Adam adjust learning rates per parameter.
2. Warmup helps avoid unstable early updates in deep models.
3. The learning-rate schedule interacts with batch size and normalization.

**Common Pitfalls**
Using the wrong learning rate can cause divergence. Removing warmup often destabilizes training for Transformers.

**Quick Example**
A training run might use Adam with a warmup schedule that increases the learning rate for the first few thousand steps, then decays it.
