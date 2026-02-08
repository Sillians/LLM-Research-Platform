# Attention Is All You Need - One-Page Summary

**Summary**
The 2017 paper "Attention Is All You Need" introduced the Transformer, a sequence-to-sequence model that replaces recurrence and convolution with attention. Each token can directly attend to every other token, enabling parallel training, better long-range dependency modeling, and state-of-the-art machine translation results at the time. The architecture quickly became the foundation for modern large language models.

**Core Contributions**
1. A fully attention-based encoder-decoder architecture.
2. Scaled dot-product attention and multi-head attention as core building blocks.
3. Positional encodings to represent order without recurrence.
4. Strong translation results with faster training versus RNN and CNN baselines.

**Architecture at a Glance**
The Transformer has two stacks.
1. Encoder: self-attention + feed-forward layers produce contextualized inputs.
2. Decoder: masked self-attention + encoder-decoder attention + feed-forward layers generate outputs autoregressively.

Each sub-layer uses residual connections, layer normalization, and dropout.

**Key Equations**
Scaled dot-product attention:

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

Position-wise feed-forward:

```
FFN(x) = max(0, x W1 + b1) W2 + b2
```

Positional encoding (sinusoidal):

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Model Configurations**
1. `Transformer Base`: 6 layers, `d_model=512`, `d_ff=2048`, 8 heads.
2. `Transformer Big`: 6 layers, `d_model=1024`, `d_ff=4096`, 16 heads.

**Training and Results**
1. Evaluated on WMT 2014 English-to-German and English-to-French translation.
2. Used Adam with a warmup learning-rate schedule, label smoothing, dropout, and BPE tokenization.
3. Achieved state-of-the-art BLEU while training faster than prior recurrent and convolutional approaches.

**Why It Mattered**
1. Parallelism: self-attention eliminates sequential dependency during training.
2. Long-range modeling: attention creates direct paths between any token pair.
3. Scalability: the architecture scales smoothly with model size and data.
4. Generality: the same backbone works across NLP, vision, speech, and multimodal tasks.

**Limitations**
1. Quadratic time and memory in sequence length due to attention.
2. Fixed positional encodings are less flexible than later alternatives.
3. Very long sequences remain challenging without specialized attention variants.
