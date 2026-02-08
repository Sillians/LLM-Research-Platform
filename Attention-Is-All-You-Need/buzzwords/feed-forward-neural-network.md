# Feed-Forward Neural Network

A feed-forward neural network is a network without cycles, where information flows from input to output through one or more layers.

**In the Transformer**
Each layer includes a position-wise feed-forward network that applies the same two-layer MLP to every token independently.

**Why It Matters**
The feed-forward block adds nonlinearity and capacity beyond attention, enabling richer transformations of token representations.

**Key Points**
1. It is applied independently to each token position.
2. The hidden layer expands to d_ff and projects back to d_model.
3. The original paper uses a ReLU activation.

**Common Pitfalls**
Forgetting that the same weights are reused for all positions can lead to incorrect implementations. Using the wrong activation changes behavior.

**Quick Example**
Each token vector is passed through a two-layer MLP that expands and then contracts its dimensionality.
