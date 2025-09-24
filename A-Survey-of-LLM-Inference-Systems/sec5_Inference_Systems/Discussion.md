# **5.3 Discussion**

### **1. General vs. Specialized Systems**

* **General-purpose inference systems:**
  Examples → Clipper, TensorFlow Serving, TensorRT, Clockwork, Hugging Face Accelerate.

  * Pros: Broad applicability, easier integration.
  * Cons: Underperform for LLMs due to lack of specialized optimizations.
* **Dedicated LLM inference systems:**

  * Optimized specifically for throughput, latency, and memory.
  * Widely adopted innovations include:

    * **Paged attention**
    * **Continuous batching with chunked prefills**
    * **MLQ scheduling** (multi-level queue)
    * **Asynchronous cache recovery over tiered storage**



### **2. System Architectures**

* **Multi-replica systems:**

  * Require **load balancing** and **distributed cache management** (hot cache replicas, cache transfer mechanisms).
* **Single-replica systems:**

  * Simpler, no distributed overhead.
  * Often better if multi-replica complexity isn’t needed.
* **Disaggregated systems:**

  * Prefill and decode stages run on **separate workers**.
  * Provide **more flexible scaling** than monolithic setups.
  * Mixed worker pools make them especially effective.
* **Serverless systems (e.g., DeepFlow):**

  * Best for **shared infrastructure**.
  * Challenges: cold starts, stateless workers.
  * Mitigation: prewarmed resource pools, persistent allocation across requests, centralized cache sharing.



### **3. Quantization Tradeoffs**

* **Compression vs. accuracy** depends on:

  * Model architecture.
  * Task type (interactive vs. batch).
* **Optimal quantization scheme** is **application-dependent**.
* Future direction: inference systems may differentiate by **adaptive quantization schemes**.



### **4. Adaptivity & Elasticity**

* **Why needed:**
  Workloads are inherently unpredictable (different request lengths, KV cache growth, interactive vs. batch).
* **Adaptive strategies:**

  * Constrained generation (limit output scope).
  * Workload disaggregation (interactive vs. non-interactive).
  * **Dynamic batch sizing**.
  * **Elastic resource provisioning**.
* **Research directions & examples:**

  * **SplitWise, TetriInfer:** dynamically adjust ratio of prefill and decode workers.
  * **iServe:** lightweight fingerprint models for quick configuration assessment.
  * **Elastic inference systems:** adapt resources in real time to workload shifts.



**Bottom line:**
Dedicated LLM inference systems are evolving toward convergence: combining paged attention, batching, adaptive load balancing, and elastic disaggregation. Future differentiators will likely emerge in **quantization schemes** and **adaptive workload management**.

---
