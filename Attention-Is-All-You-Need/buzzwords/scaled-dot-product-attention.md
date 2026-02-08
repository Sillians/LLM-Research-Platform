# Scaled Dot-Product Attention

Scaled dot-product attention computes similarity between queries and keys using a dot product, scales by the square root of the key dimension, and applies softmax to produce weights.

**In the Transformer**
Every attention head uses scaled dot-product attention as its basic computation.

**Why It Matters**
The scaling term prevents the dot products from growing too large, which would otherwise push softmax into saturated regions and slow learning.

**Key Points**
1. Similarity is computed as Q K^T.
2. Scaling by sqrt(d_k) stabilizes gradients.
3. The output is a weighted sum of values V.

**Common Pitfalls**
Omitting the scaling factor can cause training instability. Dimension mismatches between Q, K, and V are also common implementation errors.

**Quick Example**
If two tokens are highly similar, their dot product is large, the softmax weight is high, and the value vector for that token contributes more to the output.
