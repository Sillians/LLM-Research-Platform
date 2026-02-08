# Attention

Attention is a mechanism that computes a weighted mixture of information based on relevance. Given a query, it scores how much each key matches, then uses those scores to blend the corresponding values. This turns raw inputs into context-aware representations.

**In the Transformer**
Attention is the core computation in both encoder and decoder layers. It is used for self-attention (within a sequence) and cross-attention (from decoder to encoder outputs).

**Why It Matters**
Attention provides direct, content-based connections between tokens, which shortens the path between distant words and enables strong long-range dependency modeling.

**Key Points**
1. Attention weights are produced by a similarity function and normalized by softmax.
2. The output is a weighted sum of value vectors, not a single selected token.
3. The mechanism is fully differentiable and parallelizable.

**Common Pitfalls**
Attention weights are not always faithful explanations. They can be useful signals, but they do not guarantee causal influence.

**Quick Example**
In the sentence "The animal chased the ball because it was excited," attention can help the token "it" focus on "animal" instead of "ball.
