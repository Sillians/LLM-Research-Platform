# Self-Attention

Self-attention is attention where the queries, keys, and values all come from the same sequence. Each token can directly attend to every other token, producing a contextualized representation for each position.

**In the Transformer**
Self-attention is used in every encoder layer. In the decoder it is masked to prevent access to future tokens during autoregressive generation.

**Why It Matters**
It enables each token to incorporate information from the entire sequence in a single layer, improving long-range dependency modeling and supporting parallel training.

**Key Points**
1. Each token computes attention over all tokens, including itself.
2. Positional encodings supply order information that attention alone lacks.
3. Masking is required in the decoder to avoid information leakage.

**Common Pitfalls**
Confusing self-attention with cross-attention can lead to incorrect model wiring. Also, forgetting the causal mask in the decoder breaks autoregressive training.

**Quick Example**
In "The keys on the table are missing," the token "keys" can attend to "table" to encode location context.
