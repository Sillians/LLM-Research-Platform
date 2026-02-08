# Layer Normalization

Layer normalization normalizes the activations across the feature dimension for each token independently. It stabilizes training by keeping activations in a consistent range.

**In the Transformer**
The original Transformer applies layer normalization after adding the residual connection in each sub-layer.

**Why It Matters**
It reduces sensitivity to initialization and learning rate, improving stability for deep architectures.

**Key Points**
1. Normalization is performed per token, not across the batch.
2. It uses learnable scale and bias parameters.
3. It is applied repeatedly throughout the network.

**Common Pitfalls**
Confusing layer normalization with batch normalization can lead to incorrect expectations. Pre-norm and post-norm variants behave differently.

**Quick Example**
Each token vector is scaled to have consistent mean and variance before passing to the next layer.
