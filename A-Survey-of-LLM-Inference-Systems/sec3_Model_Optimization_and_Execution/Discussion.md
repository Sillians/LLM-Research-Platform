# **3.4 Discussion**


### **1. Dynamic Batching vs. Memory Pressure**

* **Large batch sizes** → higher throughput but risk **memory preemptions**.
* **Small batch sizes** → safer but underutilize hardware.
* Common strategy:

  * Use a **token budget** (upper bound on active tokens per batch).
  * Batches are dynamically sized up to this budget (balancing TTFT – *time to first token* – and TBT – *time between tokens*).
  * Example: **Sarathi-Serve** tunes token budgets offline via stress tests.

⚖️ **Trade-off:** throughput vs. memory safety.



### **2. Load Prediction Approaches**

The hardest problem is **predicting output length (decoding rounds)** and **memory load fluctuations**.
Several approaches exist:

* **Relative ranking prediction (OPT-125M)**

  * Instead of predicting exact output length, predict **relative rank** among prompts.
  * More robust than regression.

* **Range prediction (DistilBERT)**

  * Predict which **interval/range** the output length falls into.
  * Avoids precision errors of exact prediction.

* **MLP-based prediction**

  * Use **LLM internal activations** as features.
  * Train a small **MLP** to predict output length/memory load.

* **Prompt-internal prediction (LLM self-prediction)**

  * Append **special prefixes** to the prompt, ask the LLM itself to predict length.



### **3. Beyond Length Prediction – Full Load Modeling**

Output length alone is not enough; systems must consider **memory fluctuations** from:

* **KV cache growth**.
* **Cache transfers** (offloading/restoring).
* **Memory reclamation rates** (how fast freed memory can be reused).

Examples:

* combines **memory usage prediction** + **reclamation rates** for aggressive estimation.
* **Mooncake** includes **KV cache transfer overheads** in its load predictions.



### **4. Co-Design with Applications**

* In certain **restricted applications**, inference systems can be **co-designed with the frontend**:

  * Example: enforce constraints on output length (fixed ranges, capped completions).
  * This makes length prediction much more accurate and reduces scheduling complexity.



### **5. Connection to Scheduling & Load Balancing**

* **For SJF Scheduling:** predicted **output length** directly determines priority.
* **For Load Balancing:** predictions must factor in not just length, but also **cache dynamics, preemption risks, and memory fluctuations**.



**Key Takeaway:**
Load prediction remains one of the *most difficult unsolved problems* in LLM inference. Approaches range from **LM-based ranking/range prediction** to **MLP models on LLM activations**, but the inherent uncertainty of generation makes it imperfect. Practical systems therefore combine **prediction models + dynamic adjustments + cache-aware heuristics**.

---


