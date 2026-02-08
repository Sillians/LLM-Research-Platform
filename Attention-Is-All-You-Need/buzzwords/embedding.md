# Embedding

An embedding maps discrete symbols, such as tokens, into continuous vectors. Embeddings allow the model to learn similarity and structure in a dense space.

**In the Transformer**
Input tokens are converted into embeddings and then combined with positional encodings. The embedding matrix is learned during training.

**Why It Matters**
Embeddings provide a compact, trainable representation of vocabulary items and are the foundation for all downstream transformations.

**Key Points**
1. Embeddings are learned lookup vectors for each token.
2. They are often scaled by sqrt(d_model) in the Transformer.
3. Some models share the input embedding matrix with the output softmax weights.

**Common Pitfalls**
A mismatched vocabulary or tokenization scheme can silently degrade performance. Very small embedding sizes can limit model capacity.

**Quick Example**
The token "cat" might map to a 512-dimensional vector that is close to the vector for "dog.
