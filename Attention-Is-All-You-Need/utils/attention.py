"""Attention mechanisms implemented from scratch using NumPy.

This module provides scaled dot-product attention and multi-head attention.
It is designed for clarity over performance and does not implement backprop.
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def make_causal_mask(seq_len: int) -> np.ndarray:
    """Return a boolean causal mask of shape (1, 1, seq_len, seq_len).

    True means the position is allowed to attend; False means masked.
    """
    upper = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    allowed = ~upper
    return allowed[None, None, :, :]


def _to_additive_mask(mask: np.ndarray | None, dtype: np.dtype) -> np.ndarray | None:
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.dtype == np.bool_ or mask.dtype == bool:
        return np.where(mask, 0.0, -1e9).astype(dtype)
    return mask.astype(dtype)


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute scaled dot-product attention.

    Args:
        q: Queries, shape (batch, heads, seq_q, d_head)
        k: Keys, shape (batch, heads, seq_k, d_head)
        v: Values, shape (batch, heads, seq_k, d_head)
        mask: Optional mask broadcastable to (batch, heads, seq_q, seq_k).

    Returns:
        context: shape (batch, heads, seq_q, d_head)
        weights: attention weights, shape (batch, heads, seq_q, seq_k)
    """
    d_head = q.shape[-1]
    scores = np.matmul(q, np.swapaxes(k, -2, -1)) / np.sqrt(d_head)
    mask_add = _to_additive_mask(mask, scores.dtype)
    if mask_add is not None:
        scores = scores + mask_add
    weights = softmax(scores, axis=-1)
    context = np.matmul(weights, v)
    return context, weights


class MultiHeadAttention:
    """Multi-head attention implemented with NumPy."""

    def __init__(self, d_model: int, num_heads: int, seed: int = 0, rng=None) -> None:
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        if rng is None:
            rng = np.random.default_rng(seed)

        scale = 1.0 / np.sqrt(d_model)
        self.W_q = rng.normal(scale=scale, size=(d_model, d_model))
        self.W_k = rng.normal(scale=scale, size=(d_model, d_model))
        self.W_v = rng.normal(scale=scale, size=(d_model, d_model))
        self.W_o = rng.normal(scale=scale, size=(d_model, d_model))

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        batch, seq_len, _ = x.shape
        x = x.reshape(batch, seq_len, self.num_heads, self.d_head)
        return np.transpose(x, (0, 2, 1, 3))

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        batch, heads, seq_len, d_head = x.shape
        x = np.transpose(x, (0, 2, 1, 3))
        return x.reshape(batch, seq_len, heads * d_head)

    def forward(
        self,
        x_q: np.ndarray,
        x_kv: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute multi-head attention.

        Args:
            x_q: Query input, shape (batch, seq_q, d_model)
            x_kv: Key/value input, shape (batch, seq_k, d_model). If None, uses x_q.
            mask: Optional mask broadcastable to (batch, heads, seq_q, seq_k).

        Returns:
            output: shape (batch, seq_q, d_model)
            weights: shape (batch, heads, seq_q, seq_k)
        """
        if x_kv is None:
            x_kv = x_q

        q = x_q @ self.W_q
        k = x_kv @ self.W_k
        v = x_kv @ self.W_v

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        context, weights = scaled_dot_product_attention(q, k, v, mask=mask)
        merged = self._merge_heads(context)
        output = merged @ self.W_o
        return output, weights
