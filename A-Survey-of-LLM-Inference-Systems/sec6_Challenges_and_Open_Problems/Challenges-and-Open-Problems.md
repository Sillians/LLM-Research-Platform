# **6. Challenges and Open Problems**


### **1. Request Processing Operators & Algorithms**

* **LLMs beyond natural language** (e.g., multi-modal, robotics, bioinformatics) → require:

  * New **operators** tailored to different attention patterns.
  * New **sequence generation techniques** optimized for non-text tasks.
* **Opportunity:** Development of domain-specific inference operators.



### **2. Model Execution**

* **Kernels:**
  New operators/algorithms → require new optimized kernels for GPUs, TPUs, and emerging hardware.
* **Load Prediction:**
  Remains an unsolved challenge for:

  * Job prioritization (which request should run first).
  * Load balancing (where requests should be routed).
  * Accurate prediction = key for reducing tail latency.



### **3. Memory Management**

* **Quantization Complexity:**

  * Too many quantization schemes exist (int4, int8, mixed precision, etc.).
  * Open challenge: **systematic framework** for when/what/how to quantize.
* **Novel memory hierarchies:**

  * Example: **pyramidal caches** (tiered memory with GPU HBM, CPU DRAM, SSD).
  * **Entry merging:** combining similar cache entries to reduce memory footprint.
* **Elasticity need:**

  * Systems must **adaptively manage limited memory/hardware** as workloads change.



### **4. Test-Time Scaling**

* Traditional focus = minimizing latency.
* **New paradigm:** maximize **inference quality under a fixed latency budget**.
* Example: Deep research systems accept **longer latency for higher-quality outputs** (e.g., web analysis).
* Implication: Future systems may optimize differently for **quality-sensitive vs. latency-sensitive applications**.



### **5. Adapter Loading (LoRA & Profile Switching)**

* Many modern LLMs support **dynamic adapter swapping** (LoRA, PEFT).
* **Challenge:** Efficiently load/swap adapters at inference without high overhead.
* Open question: how to design inference runtimes for fast adapter switching.



### **6. Specialized LLMs**

* **Mobile LLMs** → optimized for resource-constrained devices.
* **Multimodal LLMs** → text + vision/audio.
* **Robotic LLMs** → real-world physical interactions.
* Challenge: inference systems must handle **heterogeneous requirements**:

  * Low power (mobile).
  * Cross-modality (multi-modal).
  * Real-time responsiveness (robotics).



### **7. Hardware Co-Design**

* New accelerators (AI chips, NPUs, memory-optimized devices) → add complexity.
* Challenge: **job scheduling and load balancing** must account for diverse hardware characteristics.
* Early progress: specialized scheduling techniques tuned to specific hardware backends.



**Overall Takeaway:**
Future inference systems must evolve beyond latency optimization to **adaptive, elastic, and domain-aware architectures**. Key open challenges include:

* Better **load prediction**.
* Systematic **quantization frameworks**.
* Efficient **adapter loading**.
* **Specialized inference strategies** for new LLM types and heterogeneous hardware.

---
