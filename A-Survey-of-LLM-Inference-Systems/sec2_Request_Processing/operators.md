# **2.2 Operators**

## **Inference Operator Design: Performance vs. Accuracy**

Large Language Model (LLM) inference is dominated by three key operators: **Attention**, **Feed-Forward Networks (FFN)**, and **Token Sampling**. Improving performance while preserving accuracy motivates specialized variants of these operators.


### **1. Attention Operators**

#### Standard Multi-Head Attention (MHA)

Given input $X \in \mathbb{R}^{n \times d}$ with sequence length $n$ and hidden dimension $d$, each head applies:

$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$

Attention weights per head:

$A = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d\_k}}\right)V$

where $d_k = d / h$, $h$ is the number of heads.

**Cost**:

* Memory: $O(n^2)$ for storing $QK^\top$
* Compute: $O(h \cdot n^2 \cdot d_k) = O(n^2 d)$

This quadratic cost dominates long-sequence inference.



#### Multi-Query Attention (MQA) / Grouped-Query Attention (GQA)

MQA: all heads **share $K, V$**.

GQA: groups of heads share $K, V$.

* Compute: still $O(n^2 d)$
* Memory: reduced from $O(hn^2)$ to $O(g n^2)$, where $g \ll h$.

This reduces **KV cache size** in autoregressive decoding:

* MHA KV cache: $O(h \cdot n \cdot d_k)$

* MQA KV cache: $O(n \cdot d_k)$



#### Sparse Attention

Restrict each token to attend to $O(\log n)$ or $O(\sqrt{n})$ neighbors instead of $O(n)$.

* Compute: $O(n \cdot k \cdot d)$ where $k \ll n$.
* Improves latency but risks lower accuracy if the sparsity pattern misses dependencies.



#### Shared Attention

Reuses cached attention outputs across multiple requests (batching for similarity).

* Effective for **retrieval-augmented generation** and multi-user systems.
* Lowers memory pressure by amortizing KV storage.



### **2. Feed-Forward Networks (FFN)**

Standard FFN in Transformers:

$\mathrm{FFN}(x) = \sigma(x W_1 + b_1)W_2 + b_2$

* Compute per token: $O(d^2)$
* Memory: $O(d^2)$ for weights



#### Mixture-of-Experts (MoE)

Split FFN into $E$ experts, each smaller. A router selects $k \ll E$ experts per token.

$\mathrm{MoE}(x) = \sum_{i \in \mathcal{S}(x)} g_i(x) f_i(x)$

where $\mathcal{S}(x)$ is the selected set of experts, $g_i(x)$ is gating weight.

* Compute per token: $O(k d^2)$
* Model capacity grows $\sim E d^2$, but per-token compute remains bounded.

**Trade-off**:

* Higher accuracy due to larger model capacity.
* Latency reduced since only a fraction of experts are active.



### **3. Token Sampling Operators**

At inference, logits $z_t$ yield probability distribution:

$p(y_t \mid y_{<t}) = \mathrm{softmax}(z_t)$



#### Stochastic Sampling

* **Top-k**: sample from $k$ most probable tokens.
* **Nucleus (top-p)**: sample from smallest set $\mathcal{V}*p$ such that $\sum*{i \in \mathcal{V}_p} p_i \geq p$.

Improves diversity → higher subjective accuracy for creative tasks.


#### Speculative Decoding

Use a small **draft model** $M_d$ to propose $m$ tokens $\hat{y}_{t+1\:t+m}$, then validate in parallel with large model $M$.

If accepted, throughput improves by factor $\approx m$.



### **System Design Implications**

* **Latency**:

  * Attention dominates *time-to-first-token (TTFT)*.
  * KV cache size affects *time-between-tokens (TBT)*.
  * MoE reduces per-token FFN cost.
  * Speculative decoding reduces end-to-end request latency.

* **Accuracy**:

  * MHA > MQA/GQA > sparse/shared (accuracy trade-off).
  * MoE improves accuracy (capacity scaling).
  * Sampling methods (nucleus, top-k) improve human-perceived output quality.


### **Complexity & Memory Summary**

| Operator             | Compute Complexity | Memory                                    | Accuracy Trade-off              |
| -------------------- | ------------------ | ----------------------------------------- | ------------------------------- |
| MHA                  | $O(n^2 d)$         | $O(h \cdot n \cdot d_k)$                              | High accuracy                   |
| MQA/GQA              | $O(n^2 d)$         | $O(g \cdot n \cdot d_k)$                              | Slightly lower accuracy         |
| Sparse Attn          | $O(n k d)$         | $O(n \cdot k)$                                  | Risk of missing dependencies    |
| MoE-FFN              | $O(k d^2)$         | $O(E d^2)$ weights, $O(k d^2)$ active     | Higher accuracy                 |
| Speculative Decoding | $\sim O(n d)$      | Dual models                               | Accuracy depends on draft model |


Operator innovations directly target bottlenecks (attention = quadratic scaling, FFN = compute-heavy, sampling = sequential bottleneck). System design balances **latency, throughput, and quality** via these specialized techniques.

---


## **2.2.1 Attention**









