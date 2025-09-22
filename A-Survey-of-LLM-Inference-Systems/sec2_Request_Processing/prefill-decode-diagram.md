![Prefill and Decode](../../images/prefill-decode.png)

This diagram is illustrating the **inference workflow of a transformer-based LLM**, split into **(a) Prefill** and **(b) Decode** phases. Let’s break it down step by step.


## **1. Core Idea**

LLM inference happens in two stages:

* **Prefill**: Process the entire input sequence at once to build key/value (KV) representations for attention.
* **Decode**: Generate output tokens one at a time using cached KV vectors, so past computations aren’t repeated.



## **2. Components in the Diagram**

### **Embeddings**

* **Input Embedding (light green)**: Raw token embeddings (“The”, “cat”, “sat”, “on”, “the”).
* **Contextualized Embedding (green)**: Embeddings updated after passing through layers (attention + feed-forward).

### **Attention Mechanism**

* **Query Vector (blue stripes)**: Represents the current token’s request for context.
* **Key/Value Vectors (dark blue)**: Represent stored contextual information about tokens.
* **Delta Vector (orange)**: The change/update applied to embeddings after attending to context (captures new information gained).

### **KV Cache (Decode stage only)**

* Stored key/value vectors from the Prefill stage.
* Prevents recomputation — when decoding new tokens, the model reuses past context.

### **Other Blocks**

* **Linear Transform**: Projects embeddings into queries, keys, and values.
* **Add & Normalize**: Residual connection + layer normalization.
* **Feed-Forward**: Position-wise transformations applied to embeddings.
* **Token Sampler**: Chooses the next token based on probabilities from the final embedding.



## **3. Workflow by Stage**

### **(a) Prefill Stage**

1. Input sequence (“The cat sat on the”) is embedded (light green).
2. A **linear transform** projects embeddings into query, key, and value vectors.
3. Attention computes relationships between tokens: each token attends to all previous ones.
4. The **delta vector (orange)** represents how embeddings are updated after attention.
5. After **Add & Normalize + Feed-Forward + Add & Normalize**, the contextualized embeddings (dark green) are output.
6. Token sampler predicts the **first output token** (e.g., “mat.”).
7. Keys and values are **cached** for future reuse.


### **(b) Decode Stage**

1. The newly generated token (“mat.”) is embedded (green).
2. Linear transform projects it into query, key, and value vectors.
3. **KV Cache** (from Prefill) supplies the stored context (“The cat sat on the”).
4. Attention uses the new query vector against cached KVs, producing a delta vector.
5. Embedding is updated via normalization + feed-forward + normalization.
6. Token sampler predicts the next token (e.g., “It”).
7. Cycle repeats until stopping condition (e.g., end of sequence).


## **4. Why This Matters**

* **Prefill** handles full input context (expensive, \$O(n^2)\$ cost due to full attention).
* **Decode** reuses cached context, so each step is \$O(n)\$ (only new token attends to past).
* **Delta Vectors** track incremental updates per token.
* This split is central to **efficient autoregressive inference** in LLMs.


### **In short:**

* **Prefill** = process input all at once, cache KV.
* **Decode** = generate tokens step-by-step using cached KV.
* **Delta Vector** = the intermediate update applied to embeddings during attention.


---


## **Delta Vectors in LLM Inference: A System Design Perspective**


### **1. Attention Background**

For a transformer layer, given a sequence of $t$ tokens with hidden states $H \in \mathbb{R}^{t \times d}$, the attention mechanism computes:

$$
Q = H W_Q, \quad K = H W_K, \quad V = H W_V
$$

$$
\mathrm{Attn}(H) = \mathrm{softmax}\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V
$$

* $d$: hidden dimension
* $d_{k}$: key/query dimension
* Complexity (naïve full recomputation):

$$
\mathcal{O}(t^2 d)
$$



### **2. Incremental Decoding and KV Cache**

During **autoregressive decoding**, at step $t+1$ we add one new token $x_{t+1}$.

* Instead of recomputing $Q,K,V$ for the entire prefix, we **cache past $(K_{1:t}, V_{1:t})$**.
* At decode step $t+1$:

$$
Q_{t+1} = h_{t+1} W_Q, \quad K_{t+1} = h_{t+1} W_K, \quad V_{t+1} = h_{t+1} W_V
$$

$$
\mathrm{Attn}(h_{t+1}) = \mathrm{softmax}\left( \frac{Q_{t+1} K_{1:t+1}^\top}{\sqrt{d_k}} \right) V_{1:t+1}
$$

* Complexity per step:

  $$
  \mathcal{O}(td)
  $$

  vs full recompute: $\mathcal{O}(t^2 d)$.



### **3. Delta Vector Formulation**

Define the hidden state at step $t$ as $h_t \in \mathbb{R}^d$.
The *full recompute update* is:

$$
h_{1:t} = f(h_{1:t-1}, x_t)
$$

Instead, the **delta vector** $\Delta h_t$ encodes the *incremental change*:

$$
h_t = h_{t-1} + \Delta h_t
$$

where $\Delta h_t$ is derived only from the **new token’s contribution** via $Q_t, K_t, V_t$.

In practice:

* **Prefill phase**: compute $h_{1:T}$ for input context in bulk.
  
* **Decode phase**: each new token contributes a **delta vector** $\Delta h_{t+1}$ that updates cached states.



### **4. Memory Cost Analysis**

* **KV Cache**:
  For $L$ layers, sequence length $t$, and head dimension $d_k$:

  $$
  \mathrm{Memory} = \mathcal{O}(L \cdot t \cdot d_k)
  $$

* **Delta Vector Storage**:
  At step $t+1$, only $\Delta h_{t+1} \in \mathbb{R}^d$ is computed and applied, avoiding reallocation of $h_{1:t}$.

* Thus, **marginal memory per step** is:

  $$
  \mathcal{O}(d)
  $$

instead of copying $\mathcal{O}(t d)$.



### **5. Computational Complexity**

* **Naïve recomputation**: $\mathcal{O}(t^2 d)$

* **KV cache without delta abstraction**: $\mathcal{O}(td)$ per step

* **Delta vector update abstraction**:

  * Prefill: $\mathcal{O}(T^2 d)$ once for context length $T$
  * Decode: $\mathcal{O}(td)$ per token, but structured as $\Delta h_t$ updates, enabling **system-level batching/fusion**.



### **6. System Design Implications**

* **Pipeline separation**:

  * Prefill = bulk parallelism
  * Decode = incremental updates (delta vectors)
    → clean modularity for GPU/TPU scheduling.

* **Memory bandwidth**:
  Delta vectors reduce memory movement compared to re-copying full hidden states.

* **Parallelism opportunities**:

  * **Speculative decoding**: propose multiple $\Delta h_t$ in parallel, then roll back/merge.
  * **Streaming inference**: apply $\Delta h_t$ as data arrives (low latency).
  * **Hybrid batching**: prefill + multiple delta updates fused into kernels.



### **7. Summary Table**

| Mode                | Complexity per step              | Memory per step           | Notes                         |
| ------------------- | -------------------------------- | ------------------------- | ----------------------------- |
| Full recomputation  | $\mathcal{O}(t^2 d)$           | $\mathcal{O}(t d)$      | Impractical for long contexts |
| KV cache (baseline) | $\mathcal{O}(td)$              | $\mathcal{O}(L t d_k)$ | Standard inference            |
| Delta vector update | $\mathcal{O}(td)$ (structured) | $\mathcal{O}(d)$        | Enables system optimizations  |



**In essence:** The **delta vector** is a *computational abstraction* for the incremental hidden state update during decoding. It doesn’t change asymptotic complexity, but it *changes the system design space* — making batching, fusion, speculative decoding, and memory-efficient scheduling possible.

---



