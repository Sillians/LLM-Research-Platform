# Residual Connection

A residual connection adds a layer's input to its output, forming y = x + f(x). This helps deep networks train more reliably.

**In the Transformer**
Each attention and feed-forward sub-layer is wrapped by a residual connection, followed by layer normalization.

**Why It Matters**
Residual paths improve gradient flow and reduce the risk of vanishing gradients, which is crucial for stacking many layers.

**Key Points**
1. Residuals allow layers to learn refinements instead of complete transformations.
2. They stabilize optimization in deep models.
3. They are typically paired with normalization.

**Common Pitfalls**
Dimension mismatches can prevent addition. Skipping normalization can make training unstable in deep stacks.

**Quick Example**
If a sub-layer learns nothing useful early in training, the residual path still preserves the input signal.
