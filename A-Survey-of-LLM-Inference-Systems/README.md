# Survey of LLM Inference Systems

This file contains a summary and discussion of specialized **large language model (LLM) inference systems**, based on the research paper  [**"A Survey of LLM Inference Systems."**](https://arxiv.org/pdf/2506.21901v1)


## Overview

In recent years, specialized LLM inference systems such as **vLLM**, **SGLang**, **Mooncake**, and **DeepFlow** have emerged, alongside the rapid adoption of services like **ChatGPT**. These developments are driven by the **unique autoregressive nature of LLM request processing**, which requires new techniques to achieve **high performance** while maintaining **inference quality** under **high-volume and high-velocity workloads**.

Although many techniques exist in the literature, few have been analyzed under the framework of a complete inference system, and comprehensive comparisons between systems remain limited. This survey addresses that gap.


## Topics Covered

The survey explores inference techniques across different layers of the stack:

1. **Request Processing**
   - Operators and algorithms for efficient handling of LLM requests.

2. **Model Optimization and Execution**
   - Kernel design  
   - Batching strategies  
   - Scheduling techniques  

3. **Memory Management**
   - Paged memory  
   - Eviction and offloading methods  
   - Quantization  
   - Cache persistence  


## Key Insights

- Most techniques fundamentally rely on:
  - **Load prediction**  
  - **Adaptive mechanisms**  
  - **Cost reduction**  

- These approaches help overcome the challenges introduced by **autoregressive generation** and enable scalable, efficient inference.


## System Designs

The paper also examines how these techniques combine to form complete inference systems:

- **Single-replica inference systems**  
- **Multi-replica inference systems**  
- **Disaggregated inference systems** (offering finer control over resource allocation)  
- **Serverless systems** (scalable, on-demand inference)  


---
