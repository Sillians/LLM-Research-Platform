# Dropout

Dropout randomly zeros a fraction of activations during training. It is a simple and effective regularization technique.

**In the Transformer**
Dropout is applied to attention weights, residual connections, and feed-forward activations in the original model.

**Why It Matters**
It reduces overfitting by preventing co-adaptation of features and encourages more robust representations.

**Key Points**
1. Dropout is active only during training.
2. The remaining activations are scaled to keep expected values consistent.
3. It is often combined with other regularization methods.

**Common Pitfalls**
Forgetting to disable dropout at inference leads to degraded performance. Too much dropout can underfit.

**Quick Example**
With 0.1 dropout, about 10 percent of activations are zeroed each training step.
