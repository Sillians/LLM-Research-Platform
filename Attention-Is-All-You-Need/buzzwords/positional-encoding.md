# Positional Encoding

Positional encoding injects information about token order into the model. Without it, attention is permutation-invariant and cannot tell one ordering from another.

**In the Transformer**
The paper uses fixed sinusoidal positional encodings added to token embeddings before entering the encoder and decoder stacks.

**Why It Matters**
Order is essential for language. Positional encoding lets the model distinguish "dog bites man" from "man bites dog."

**Key Points**
1. Position vectors are added to token embeddings, not concatenated.
2. Sinusoidal encodings allow extrapolation to longer sequences.
3. Learned positional embeddings are a common alternative.

**Common Pitfalls**
Forgetting positional encodings severely degrades performance. Mismatched maximum sequence length can cause indexing errors.

**Quick Example**
Two sequences with identical tokens but different order become distinguishable only after positional encoding is applied.
