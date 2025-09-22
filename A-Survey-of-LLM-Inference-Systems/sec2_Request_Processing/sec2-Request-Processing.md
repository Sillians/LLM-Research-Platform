# Request Processing

Transformer-based language models operate over a discrete set of tokens, generating one output token at a time in response to a sequence of input tokens. The process involves mapping tokens into a high-dimensional vector space, applying linear transformations to create contextualized embeddings, and then selecting the next output token.

This workflow relies heavily on operators such as **attention**, **feed-forward layers**, and **token sampling**, which form the backbone of LLM inference. However, the high computational costs of attention and feed-forward operations, combined with the need to produce semantically coherent and contextually appropriate text, have motivated extensive research into more efficient operator designs.

Since tokens can only be generated sequentially, producing long outputs requires multiple rounds of model execution, where each output token depends sensitively on the previous inputs and outputs. This has led to the development of specialized **sequence generation techniques** aimed at improving efficiency and quality.

System designers face the challenge of evaluating these diverse request processing methods, balancing their trade-offs, and aligning them with broader system goals. The choices made at this level influence both the efficiency of inference systems and the quality of generated outputs.