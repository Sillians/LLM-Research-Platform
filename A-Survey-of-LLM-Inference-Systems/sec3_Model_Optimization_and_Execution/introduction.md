# **3. Model Optimization and Execution**

This section focuses on how LLM inference systems exploit **GPU parallelism** and manage execution efficiency through **kernel design, batching, and scheduling**.


## 1. **Why GPUs / Parallel Processors?**

* **CPUs** are general-purpose but slower for **data-parallel operations**.
* **GPUs (and TPUs, AI accelerators)** excel at extreme parallelism, making them the core for LLM inference.
* To fully utilize them, systems apply specialized techniques across three layers: **kernel design, batching, and scheduling**.



## 2. **Kernel Design (Section 3.1)**

* **Goal**: Write optimized GPU kernels for inference operators (attention, FFN, normalization, etc.).
* Optimizations:

  * Memory access patterns (avoid cache thrashing).
  * Operator fusion (merge multiple small kernels into one).
  * Quantization-aware kernels (8-bit, 4-bit inference).
  * Exploiting sparsity (skip unnecessary compute).



## 3. **Request Batching (Section 3.2)**

* **Goal**: Aggregate multiple inference requests to maximize GPU utilization.
* Benefits: Saturates GPU compute capacity.
* Challenges:

  * **Stragglers**: A slow request can delay an entire batch.
  * **KV cache growth**: Each request maintains a key-value cache for attention. Many requests = high memory use.
* Techniques:

  * Dynamic batching (adaptive group formation).
  * Cache-aware batching (limit batch size based on available memory).



## 4. **Request Scheduling (Section 3.3)**

* **Goal**: Efficiently distribute requests across compute resources.
* Considerations:

  * **Job prioritization**: High-priority tasks (e.g., chatbot responses) get faster allocation.
  * **Load balancing**: Avoid GPU hotspots.

* **Challenge**: Need to predict **memory cost** and **execution rounds** for each request.

  * Hard because sequence length/output length is unknown.
* **Mitigation strategies**:

  * Paged attention (allocate memory dynamically).
  * Job rebalancing (migrate requests mid-execution).



**Big Picture:**
The execution layer is about **squeezing maximum throughput** out of GPUs while **managing memory pressure**.

* **Kernels** optimize individual operators.
* **Batching** optimizes across requests.
* **Scheduling** optimizes across GPUs/nodes.

---
