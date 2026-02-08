# Multi-Layer Perceptron

A multi-layer perceptron (MLP) is a feed-forward network with at least one hidden layer and a nonlinear activation function.

**In the Transformer**
The position-wise feed-forward sub-layer is an MLP. It operates independently on each token vector.

**Why It Matters**
The MLP provides non-linear transformations that attention alone cannot express, improving model capacity.

**Key Points**
1. An MLP alternates linear layers and nonlinear activations.
2. The Transformer uses a two-layer MLP in each block.
3. The MLP is shared across all positions in the sequence.

**Common Pitfalls**
Confusing the MLP with attention can lead to architectural errors. Removing the nonlinearity significantly reduces expressiveness.

**Quick Example**
An MLP transforms x into y via y = W2 * ReLU(W1 * x + b1) + b2.
