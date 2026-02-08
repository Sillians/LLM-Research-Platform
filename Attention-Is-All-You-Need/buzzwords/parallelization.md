# Parallelization

Parallelization means performing computations simultaneously instead of sequentially. In Transformers, most training operations can run in parallel across all tokens.

**In the Transformer**
Self-attention and feed-forward layers operate on the full sequence at once, enabling efficient matrix multiplication on GPUs and TPUs.

**Why It Matters**
Parallelization reduces training time and improves hardware utilization, which is a key reason Transformers scale to large datasets.

**Key Points**
1. Training is highly parallel because there is no recurrence.
2. Decoder inference is still sequential because outputs are autoregressive.
3. Memory and attention cost scale quadratically with sequence length.

**Common Pitfalls**
It is easy to overstate parallelism by ignoring autoregressive inference costs. Long sequences can still bottleneck due to O(n^2) attention.

**Quick Example**
During training, all 512 tokens can be processed in one pass instead of one token at a time.
