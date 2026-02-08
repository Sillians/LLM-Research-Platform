# Rotary Positional Embedding (RoPE)

Rotary Positional Embedding (RoPE) is a positional encoding method that injects position information directly into attention by rotating query and key vectors in a position-dependent way. Instead of adding a separate positional vector, RoPE applies a rotation to the Q and K representations so that relative positions are encoded in their dot product.

**Where It Shows Up**
RoPE is widely used in modern Transformer models, especially decoder-only LLMs. It is not part of the original "Attention Is All You Need" paper, but it is a common upgrade to the positional encoding component.

**Core Idea**
Each pair of dimensions in a vector is treated like a 2D plane. For position `p`, you rotate the query and key vectors by a fixed angle that depends on `p`. The attention score between a query at position `p` and a key at position `q` then depends on the relative offset `p - q`.

**How It Works (Simplified)**
1. Split the vector into 2D pairs: `(x0, x1)`, `(x2, x3)`, ...
2. For each pair, apply a rotation that depends on the token position.
3. Do this for both Q and K before computing attention.

This can be expressed as:

```
rot(x, p) = [x0 * cos(wp) - x1 * sin(wp), x0 * sin(wp) + x1 * cos(wp)]
```

Where `w` is a frequency that depends on the dimension index and `p` is the position.

**Why It Matters**
1. Encodes relative position information directly in attention scores.
2. Avoids adding extra vectors to embeddings.
3. Supports extrapolation to longer sequence lengths more gracefully than some fixed encodings.

**Key Points**
1. RoPE modifies Q and K, not V.
2. It is compatible with standard scaled dot-product attention.
3. Relative positions are captured through rotation, not addition.

**Common Pitfalls**
Applying RoPE to values or to only one of Q or K breaks the intended relative-position behavior. Another common issue is mismatched frequency scaling when changing model dimensions or context length.

**Quick Example**
Two tokens that are far apart receive a larger relative rotation difference, which influences their attention score even if their content vectors are similar.
