# Attention Is All You Need

Paper: [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin

## Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves `28.4` BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.



## Introduction

Sequence transduction models map an input sequence to an output sequence, where the lengths of the input and output sequences may differ. Examples of sequence transduction tasks include machine translation, speech recognition, and text summarization. The dominant approach to sequence transduction is based on encoder-decoder architectures that typically use `recurrent neural networks (RNNs)` or `convolutional neural networks (CNNs)` to build representations of the input and output sequences. These models often incorporate an attention mechanism to better connect the encoder and decoder.

In this work, we propose the Transformer, a new simple network architecture for sequence transduction that relies entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for less time. We demonstrate the effectiveness of the Transformer on two machine translation tasks: English-to-German and English-to-French. Our model achieves a BLEU score of 28.4 on the WMT 2014 English-to-German translation task and a BLEU score of 41.8 on the WMT 2014 English-to-French translation task, outperforming previous state-of-the-art models. We also show that the Transformer generalizes well to other tasks by applying it to English constituency parsing, achieving strong results with both large and limited training data.

The rest of the paper is organized as follows: 
- Section 2 describes the Transformer architecture in detail. 
- Section 3 presents the training setup and experimental results. 
- Section 4 discusses related work, and 
- Section 5 concludes the paper with future directions.


## Model Architecture

The Transformer model architecture is based on a stack of identical layers for both the encoder and decoder. Each layer consists of two main components: `multi-head self-attention` mechanisms and `position-wise fully connected feed-forward networks`. The encoder and decoder are composed of `N` layers, where `N` is a hyperparameter. The encoder processes the input sequence and generates a continuous representation, while the decoder generates the output sequence based on the encoder's representation and the previously generated tokens.


### Multi-Head Attention

The multi-head attention mechanism allows the model to jointly attend to information from different representation subspaces at different positions. It consists of several attention heads, each of which computes scaled dot-product attention. The attention mechanism takes three inputs: 
- queries (Q), 
- keys (K), and 
- values (V). 
The output of each attention head is a weighted sum of the values, where the weights are determined by the similarity between the queries and keys. The outputs of all attention heads are concatenated and linearly transformed to produce the final output.


### Position-Wise Feed-Forward Networks

Each layer in the Transformer also includes a position-wise feed-forward network, which consists of two linear transformations with a `ReLU` activation in between. This network is applied independently to each position in the sequence, allowing the model to capture complex relationships between different positions.


### Positional Encoding

Since the Transformer architecture does not use recurrence or convolution, it lacks a built-in notion of word order. To address this, we add positional encodings to the input embeddings at the bottom of the encoder and decoder stacks. These positional encodings are designed to provide information about the relative or absolute positions of the tokens in the sequence. We use `sine` and `cosine` functions of different frequencies to generate the positional encodings.


## Training Setup

We trained the Transformer model on the WMT 2014 English-to-German and English-to-French translation tasks. The training data consists of approximately 4.5 million sentence pairs for English-to-German and 36 million sentence pairs for English-to-French. We used byte-pair encoding (BPE) to preprocess the data, resulting in a shared vocabulary of 32,000 tokens for both languages. The model was trained using the Adam optimizer with a learning rate schedule that increases linearly for the first warm-up steps and then decreases proportionally to the inverse square root of the step number. We applied dropout to the attention weights and the feed-forward networks to prevent overfitting. The model was trained on 8 NVIDIA P100 GPUs for approximately 3.5 days for the English-to-French task and 12 hours for the English-to-German task.


## Experimental Results

We evaluated the performance of the Transformer model using the BLEU score, a standard metric for machine translation quality. On the WMT 2014 English-to-German translation task, the Transformer achieved a BLEU score of 28.4, outperforming previous state-of-the-art models by over 2 BLEU points. On the WMT 2014 English-to-French translation task, the model achieved a BLEU score of 41.8, setting a new single-model state-of-the-art. We also conducted ablation studies to analyze the contributions of different components of the model, such as multi-head attention and positional encoding, to the overall performance.


## Conclusion

In this paper, we introduced the Transformer, a novel neural network architecture based solely on attention mechanisms for sequence transduction tasks. The Transformer demonstrated superior performance on machine translation tasks while being more parallelizable and requiring less training time compared to traditional RNN and CNN-based models. Our experiments showed that the Transformer achieved state-of-the-art results on the WMT 2014 English-to-German and English-to-French translation tasks. Additionally, we highlighted the model's ability to generalize to other tasks, such as English constituency parsing. Future work includes exploring the application of the Transformer architecture to other sequence transduction tasks and further optimizing the model for efficiency and performance.



## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).
- Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. In Advances in neural information processing systems (pp. 3104-3112).
- Gehring, J., Auli, M., Grangier, D., Yarats, D., & Dauphin, Y. N. (2017). Convolutional sequence to sequence learning. In Proceedings of the 34th International Conference on Machine Learning-Volume 70 (pp. 1243-1252).
- Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.