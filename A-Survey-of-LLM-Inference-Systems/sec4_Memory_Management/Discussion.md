# **4.5 Discussion: Managing KV Cache & Memory in LLM Inference**

#### **Motivation**

* **Problem**: LLMs contain billions of parameters and produce **large KV caches** during inference, stressing GPU memory.
* **Goal**: Reduce memory pressure without severely degrading accuracy.


### **Main Strategies**

1. **Paged Memory**

   * Provides **flexible memory allocation**.
   * Better than static preallocation (fixed-size blocks).
   * Adapts to varying request lengths & workloads.

2. **Eviction & Offloading**

   * Free up memory by removing or moving cache entries.
   * **Offloading to tiered storage** (CPU memory, SSDs, etc.):

     * Useful for preempted requests.
     * Avoids costly cache recomputation by asynchronously restoring data.
   * **Alternative to distributed attention** (e.g., Ring Attention) for handling long contexts.

3. **Quantization**

   * Compresses **weights and activations** to lower precision.
   * **Key tradeoff**: memory savings vs accuracy.
   * Examples:

     * : Vector-wise quantization;
       • $g > 1$ for weights (lower variance).
       • $g = 1$ for activations (higher variance).
     * : $g = 1$ for both → aggressive compression, but requires **outlier protection** for activations.
   * **Model type matters**:

     * BERT (encoder-only) tolerates aggressive quantization (low weight variance).
     * GPT (decoder-only) benefits from finer quantization (vector-wise/dimension-wise).

4. **Cache Persistence**

   * **Prefix sharing**: no accuracy loss, exact-match reuse.
   * **Selective reconstruction**: more aggressive memory savings, but may reduce accuracy.
   * Choice depends on accuracy requirements.


### **System-Level Design Considerations**

* **Constrained vs Unconstrained KV Cache**

  * **Constrained**: predictable memory usage → easier batching/scheduling, but risks accuracy loss for long sequences.
  * **Unconstrained**: higher accuracy, but harder to manage resource allocation.

* **Hybrid Approaches**

  * : Allow **memory constraints to vary across layers**:

    * Allocate **more memory to earlier layers** (where attention variance is higher).
    * Later layers get less memory.
    * Results in large savings with little quality loss.

* **Advanced Ideas**

  * **Dynamic memory allocation per request**.
  * **Entry merging**: Instead of evicting, merge similar tokens so the overall cache distribution changes minimally, while reducing cache size.



### **Key Takeaways**

* **Paged memory** = flexibility.
* **Eviction/offloading** = handle long contexts, async recovery.
* **Quantization** = huge savings, but scheme depends on model type & sensitivity.
* **Cache persistence** = balance between prefix reuse (safe) and selective reconstruction (aggressive).
* **System design** = tradeoff between **predictability (constrained)** and **accuracy (unconstrained)**.
* **Future direction** = dynamic per-layer/request allocation, token merging.

---

## **Memory Optimization in LLM Inference: Taxonomy**

### **Step 1. Define System Goal**

* **Main question:** What’s the primary constraint?

  1. **Limited GPU memory** → Optimize storage size.
  2. **Long context support** → Manage growing KV caches.
  3. **Predictable performance for batching/scheduling** → Control allocation.
  4. **Accuracy preservation** → Use safer techniques.


### **Step 2. Pick a Strategy**

#### **A. Memory Allocation Strategy**

1. **Static Preallocation**

   * Fixed blocks reserved in advance.
   * Simple but inflexible.

2. **Paged Memory (Dynamic)** ✅

   * Flexible allocation/reallocation.
   * Adapts to variable sequence lengths.
   * Preferable for modern inference systems.



#### **B. Cache Size Management**

1. **Constrained Cache**

   * Predictable memory usage.
   * Simplifies scheduling.
   * May reduce accuracy for long sequences.

2. **Unconstrained Cache**

   * Better accuracy.
   * Harder to batch/schedule.

3. **Hybrid (Layer-Wise Variable Allocation)** ✅

   * More memory in earlier layers, less in later ones.
   * Balances quality vs memory.
   * Proven effective (\[13,120]).



#### **C. Cache Optimization Techniques**

1. **Eviction & Offloading**

   * **Eviction**: Remove less important tokens.
     • Simple but risks accuracy.
   * **Offloading**: Move KV caches to CPU/SSD.
     • Good for preempted requests (async recovery).
     • Alternative to distributed attention for long contexts.

2. **Quantization**

   * Reduce KV + weight precision.
   * Schemes:

     * **Tensor-wise** (coarse, big savings, less precise).
     * **Vector-wise** (finer, parameterized by \$g\$).
     * **Dimension-wise** (finest, highest control, more complex).
   * **Model sensitivity**:

     * BERT → tolerates aggressive quantization.
     * GPT → needs vector/dimension-wise.
   * **Outlier handling**:

     * Mixed-precision preservation.
     * Outlier smoothing.

3. **Cache Persistence**

   * **Prefix Sharing** (safe, exact-match).
   * **Selective Reconstruction** (aggressive, partial recomputation, may lose quality).
   * Useful in RAG or overlapping contexts.

4. **Entry Merging (Future Technique)**

   * Instead of discarding, merge similar tokens.
   * Retains information distribution while reducing size.


#### **Step 3. Decision Tree**

```
Start
│
├── Is GPU memory severely limited?
│      ├── Yes → Apply Quantization
│      │        ├── BERT → Tensor-wise / Vector-wise (aggressive)
│      │        └── GPT → Vector-wise / Dimension-wise (with outlier protection)
│      └── No → Skip to Cache Management
│
├── Do you need long context support?
│      ├── Yes → Use Eviction or Offloading
│      │        ├── Short-term reuse → Eviction
│      │        └── Preempted / async restore → Offloading
│      └── No → Skip
│
├── Do multiple requests share prefixes (e.g., RAG)?
│      ├── Yes → Cache Persistence
│      │        ├── Accuracy critical → Prefix Sharing
│      │        └── Aggressive savings OK → Selective Reconstruction
│      └── No → Skip
│
├── Is predictable memory usage needed (batching/scheduling)?
│      ├── Yes → Constrained Cache (fixed limit)
│      └── No → Unconstrained / Hybrid Layer-Wise Allocation
│
└── Advanced (Future):
       ├── Entry Merging (accuracy-preserving compression)
       └── Dynamic per-request allocation
```



### **Key Insight**

* **Choose techniques based on bottleneck**:

  * **Storage savings** → Quantization.
  * **Long contexts** → Eviction/Offloading.
  * **Shared computation** → Cache Persistence.
  * **Predictable scheduling** → Constrained/Hybrid allocation.
* **Best practice today**:
  Combine **paged memory + vector-wise quantization (with outlier protection) + offloading (for long contexts) + prefix sharing** for maximum efficiency.

---



## Practical mapping: techniques → tools / frameworks

### 1) **Paged / page-aware allocation & cache persistence**

* **vLLM** — *paged attention, page-table memory manager, cache persistence*: implemented page-based KV cache management and cache sharing for multi-request reuse. Great example if you want page granularity and prefix sharing.

* **vAttention (GPU-native page management)** — moves some bookkeeping onto the GPU to reduce CPU↔GPU round trips; useful when you want page semantics but low overhead.

* **Custom host+GPU page table** (Triton / CUDA) — if building in-house, implement a page table on host and page-aware kernels on GPU (common pattern).

**When to use:** variable-length workloads, multi-request cache reuse, long-context systems.



### 2) **Block sharing / cache persistence (prefix / partial sharing)**

* **vLLM** — prefix sharing and cache reuse examples.

* **Serving layers built on top of vLLM / custom index** — many production systems implement radix-tree or trie indexes to look up persisted prefixes quickly.

* **Retrieval-augmented frameworks** (RAG workflows) — integrate with the above to share document-chunk caches across requests.

**When to use:** repeated system prompts, RAG-heavy workloads, multi-user systems with shared prefixes.



### 3) **Blockwise / fused kernels (attention, FFN, non-GeMM ops)**

* **FlashAttention** family — blockwise fused attention kernel (low memory footprint; avoids materializing full QKᵀ). Often first-choice for single-GPU attention speedups.

* **FasterTransformer / TensorRT-LLM** — vendor-optimized kernels and runtime for high-throughput inference (attention/FFN fused, quantized kernels).

* **FlashInfer / FlashDecoding / LeanAttention** (projects implementing variants of fused/blockwise attention).

* **Triton** — write your own fused kernels (good when you need custom masking, blockwise logic, or experiments).

* **cuBLAS/cuDNN + kernel fusion wrappers** (DeepSpeed-Inference sometimes wraps vendor kernels with fusion/graph launch).

**When to use:** maximize single-GPU throughput, reduce I/O and launch overhead.



### 4) **Distributed attention / ring / multi-GPU attention**

* **Ring-Attention / All-reduce-based distributed attention** (research + implementations): partition Q/K/V tiles across GPUs and pipeline tile exchange.

* **FasterTransformer / Alpa / DeepSpeed** (distributed inference features) — provide multi-GPU inference pipelines and partitioning primitives.

**When to use:** ultra-long contexts that do not fit in one GPU, or when you want horizontal scaling of KV cache.



### 5) **Batching strategies (continuous / dynamic batching)**

* **vLLM** — dynamic/continuous batching primitives (reform batches per decode step).

* **Orca / custom queuing layers** — implement adaptive/dynamic batching and token-budget strategies.

* **Serving frameworks (Ray Serve / FastAPI + custom batcher)** — use these for request queues + dynamic batch assembly.

* **DeepSpeed-Inference** often exposes batching configuration options for throughput/latency tradeoffs.

**When to use:** real-time serving (chatbots) where arrival patterns are bursty and you need to avoid stragglers.



### 6) **Scheduling, priority, load balancing, rebalancing**

* **vLLM + cluster orchestration (Kubernetes / Ray)** — vLLM’s design shows scheduling + eviction integration examples; pair with job orchestration for priority and rebalancing.

* **Custom schedulers** (SJF/MLQ + cache-aware heuristics) — often implemented in production on top of batcher + worker pools.

* **Alpa / Orca** — frameworks that focus on distributed workload placement and can be adapted for rebalancing logic.

**When to use:** multi-replica deployments, mixed-priority request types.



### 7) **Eviction & Offloading (tiered memory)**

* **DeepSpeed ZeRO-Offload / DeepSpeed-Inference** — offloading parameters/optimizer or activations to CPU/NVMe; similar concepts apply to moving KV cache to host/disk.
  
* **vLLM** — may offload cache pages and uses async transfer patterns.

* **Custom offload managers** (host-side code + asynchronous CUDA memcpy / RDMA) — for streaming pages in/out.

**When to use:** preemption handling, very long contexts with limited GPU RAM.



### 8) **Quantization (weights & activations / KV)**

* **bitsandbytes** — widely used 8-bit/4-bit weight quantization backends and quantized optimizers for inference; easy integration for weight compression.
  
* **NVIDIA TensorRT / FasterTransformer** — quantized kernel support on NVIDIA hardware (int8/FP16).

* **Intel / ONNX Runtime** — quantization toolchains for CPU inference.

* **Custom quantization libraries** (vector-wise / dimension-wise) — used when you need per-tensor/per-vector quantizers.

* **Hugging Face + PEFT / QLoRA workflows** — for quantized fine-tuning/inference setups (practical for model compression + fine-tuning).

**When to use:** reduce model & KV storage footprint; critical for memory-constrained inference.



### 9) **Outlier protection (mixed-precision, smoothing)**

* **Mixed-format data structures + custom kernels** — frameworks that implement mixed-precision storage (store indices of outliers + low-precision body) require specialized kernels (some production systems implement this in-house).
  
* **Techniques implemented in research libraries and custom TensorRT/Triton kernels** — outlier smoothing and compensated matmul implementations exist in research and some toolkits.

* **bitsandbytes + custom adapters** — often used with mixed-precision strategies to keep outlier ranges in higher precision.

**When to use:** aggressive quantization where a few outliers would otherwise ruin accuracy.



### 10) **Speculative decoding**

* **Speculative decoding algorithms / libraries** — some inference stacks implement draft-model + verify pipelines (often custom).

* **vLLM / DeepSpeed wrappers** — can be used to implement speculative decoding flows (draft model runs followed by verification runs using batched prefill style calls).

**When to use:** increase throughput for predictable tasks (retrieval-based answers).



### 11) **CPU/GPU kernel orchestration & launch optimizations**

* **DeepSpeed-Inference (CUDA Graphs)** — consolidate kernel launches into graphs to reduce launch overhead.
  
* **CUDA Graphs / custom CUDA streams** — general approach to reduce kernel invocation overhead.

* **Triton / custom fused kernels** — also help reduce per-op overhead.

**When to use:** reduce per-token latency and kernel launch overhead.



## Short implementation guidance / how to pick

1. **Start with fused attention + dynamic batching** (FlashAttention + vLLM-style batcher) — gives big wins immediately for single-GPU inference.

2. **If memory is the bottleneck:** add vector-wise quantization (bitsandbytes) and paged allocation (vLLM).

3. **If long contexts exceed single GPU:** consider offloading (DeepSpeed offload) or distributed attention (Ring/Alpa/FasterTransformer multi-GPU).

4. **If you serve mixed workloads:** adopt continuous batching + cache-aware scheduling + priority queues.

5. **If you need extreme throughput at moderate accuracy risk:** use MQA/GQA + MoE and speculative decoding where appropriate.

6. **Instrument heavily:** these optimizations interact (batch size ↔ KV size ↔ quantization level ↔ eviction policy), so profile and tune.

---
