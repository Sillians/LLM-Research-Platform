# **4.1 Paged-Based Memory Allocation**

### **Motivation**

* KV cache grows gradually during decoding.
* Static preallocation wastes memory because it reserves the maximum size upfront.
* Paged allocation avoids this by assigning **small blocks (pages)** as needed.
* Challenge: pages are **non-contiguous in memory**, so special management is required.



### **Memory Manager**

The **memory manager** is responsible for:

1. **Page Creation/Deletion** → allocating and freeing GPU blocks dynamically.
2. **Page Lookup** → mapping logical positions (token indices in the sequence) to physical GPU addresses.

* **Page Table**:

  * Tracks addresses and contents of each page (tokens + their positions).
  * Serves as the central map for cache lookups.
* **vLLM Design**:

  * GPU holds the KV cache.
  * CPU implements memory manager + page table.
  * Page operations require **CPU–GPU communication**, adding overhead.
* **vAttention Design**:

  * Moves memory management to the **GPU itself**, cutting communication costs.
  * Side benefit: makes non-contiguous memory appear contiguous to the kernel → allows reuse of optimized **non-paged kernels**.



### **Block Sharing**

Paged allocation also enables **block sharing**, a memory optimization where multiple requests reuse the same KV blocks.

Two approaches:

1. **Exact-Match Sharing**

   * Only the **longest common prefix** across requests is shared.
   * Example: Two requests with the same system prompt share its cached pages.
   * **Block size tradeoff**:

     * Small blocks = easier to find exact matches, but more page lookups.
     * Large blocks = fewer lookups, but harder to find matches.

2. **Partial-Match Sharing**

   * Relaxes the rule: pages can be shared even if **some tokens overlap** (not necessarily aligned or sequential).
   * More aggressive in reusing cache, but requires careful kernel handling.



### **Key Insights**

* **Paged allocation** → enables dynamic growth without waste.
* **Page-aware kernels** → handle non-contiguous blocks.
* **GPU-native management** → reduces overhead (vAttention).
* **Block sharing** → reduces redundancy by leveraging common prefixes.


This section essentially sets up the foundation for **cache persistence**, since once you can share and remap pages, you can persist KV entries across requests and sessions.

---


