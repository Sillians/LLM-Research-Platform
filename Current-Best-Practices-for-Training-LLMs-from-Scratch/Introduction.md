# **Introduction**

Although we’re only a few years removed from the transformer breakthrough, LLMs have already grown massively in performance, cost, and promise. Many teams across industries are experimenting with building LLMs, but the critical details and key decision points are often passed down by word of mouth rather than clearly documented.

The goal of this white paper is to distill the best practices for training your own LLM from scratch. We’ll cover everything from scaling and hardware to dataset selection and model training, highlighting the tradeoffs to consider and flagging potential pitfalls along the way. This is meant to be a fairly exhaustive look at the key steps and considerations you’ll face when training an LLM from scratch.

The first question you should ask yourself is whether training one from scratch is the right path for your organization. In many cases, fine-tuning or adapting existing pretrained LLMs is far more efficient due to the extreme costs of compute, data, and engineering expertise. Training from scratch typically makes sense only if you need full control over the architecture and training pipeline, possess large-scale proprietary data unavailable to others, or are targeting specialized deployment environments where off-the-shelf models are not viable, such as low-latency edge use cases or highly regulated domains.



