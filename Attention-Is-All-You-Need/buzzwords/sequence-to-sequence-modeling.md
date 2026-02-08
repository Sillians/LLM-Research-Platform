# Sequence-to-Sequence Modeling

Sequence-to-sequence modeling maps an input sequence to an output sequence, often with different lengths. It is a standard setup for tasks like translation and summarization.

**In the Transformer**
The Transformer is a sequence-to-sequence model with an encoder that reads the input and a decoder that generates the output token by token.

**Why It Matters**
It provides a general framework for many NLP tasks and makes the conditional structure of the problem explicit.

**Key Points**
1. The model estimates P(output | input) and factors it autoregressively.
2. Output tokens are generated sequentially during inference.
3. Teacher forcing is commonly used during training.

**Common Pitfalls**
Exposure bias can occur when training uses ground-truth prefixes but inference uses model outputs. Evaluation metrics may not fully capture quality.

**Quick Example**
Input: "I love apples". Output: "J'aime les pommes.
