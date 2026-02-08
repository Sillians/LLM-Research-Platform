# Regularization

Regularization refers to techniques that improve generalization by discouraging overfitting. It typically trades a small increase in training loss for better validation performance.

**In the Transformer**
The paper uses dropout and label smoothing. Other common regularizers include weight decay and early stopping.

**Why It Matters**
Large models can memorize training data. Regularization helps them learn patterns that transfer to new inputs.

**Key Points**
1. Regularization reduces variance and improves robustness.
2. It often interacts with optimizer settings and learning rate schedules.
3. Different tasks and datasets require different regularization strength.

**Common Pitfalls**
Over-regularization can harm performance. Applying multiple regularizers without tuning can lead to underfitting.

**Quick Example**
Label smoothing softens the target distribution, which can improve calibration and BLEU in translation.
