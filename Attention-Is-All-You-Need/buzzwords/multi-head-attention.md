# Multi-Head Attention

Multi-head attention runs several attention operations in parallel, each with its own learned projections of queries, keys, and values. The head outputs are concatenated and projected back to the model dimension.

**In the Transformer**
Multi-head attention is used in the encoder, decoder self-attention, and encoder-decoder attention blocks.

**Why It Matters**
Different heads can focus on different relationships, such as syntax, coreference, or positional patterns, which increases model expressivity.

**Key Points**
1. Each head uses a separate linear projection of Q, K, and V.
2. Head outputs are concatenated and linearly transformed.
3. The total compute is similar to single-head attention at the same model size.

**Common Pitfalls**
Too many heads with very small per-head dimensions can reduce capacity. Another pitfall is assuming heads always specialize without inspecting results.

**Quick Example**
One head may align verbs with their subjects while another head tracks nearby positional dependencies.
