# Encoder-Decoder Architecture

An encoder-decoder architecture splits the model into two parts: an encoder that produces contextual representations of the input and a decoder that generates the output conditioned on those representations.

**In the Transformer**
The encoder is a stack of self-attention and feed-forward layers. The decoder uses masked self-attention, then attends to encoder outputs via cross-attention.

**Why It Matters**
It cleanly separates understanding the input from generating the output, enabling flexible sequence lengths and strong alignment between source and target.

**Key Points**
1. Encoder outputs serve as keys and values for decoder cross-attention.
2. Decoder self-attention is masked to preserve autoregressive generation.
3. Both stacks use residual connections and layer normalization.

**Common Pitfalls**
Mixing up which side provides keys and values in cross-attention leads to incorrect behavior. Forgetting the decoder mask leaks future tokens.

**Quick Example**
For translation, the encoder reads the source language and the decoder produces the target language while attending to the encoder outputs.
