"""Encoder-decoder building blocks implemented from scratch using NumPy."""

from __future__ import annotations

import numpy as np

from .attention import MultiHeadAttention


class LayerNorm:
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        self.gamma = np.ones((d_model,), dtype=np.float32)
        self.beta = np.zeros((d_model,), dtype=np.float32)
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class FeedForward:
    def __init__(self, d_model: int, d_ff: int, seed: int = 0, rng=None) -> None:
        if rng is None:
            rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d_model)
        self.W1 = rng.normal(scale=scale, size=(d_model, d_ff))
        self.b1 = np.zeros((d_ff,), dtype=np.float32)
        self.W2 = rng.normal(scale=scale, size=(d_ff, d_model))
        self.b2 = np.zeros((d_model,), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = np.maximum(0.0, x @ self.W1 + self.b1)
        return hidden @ self.W2 + self.b2


class EncoderLayer:
    def __init__(self, d_model: int, num_heads: int, d_ff: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.self_attn = MultiHeadAttention(d_model, num_heads, rng=rng)
        self.ffn = FeedForward(d_model, d_ff, rng=rng)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x: np.ndarray, src_mask: np.ndarray | None = None) -> np.ndarray:
        attn_out, _ = self.self_attn.forward(x, mask=src_mask)
        x = self.norm1.forward(x + attn_out)
        ffn_out = self.ffn.forward(x)
        x = self.norm2.forward(x + ffn_out)
        return x


class DecoderLayer:
    def __init__(self, d_model: int, num_heads: int, d_ff: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.self_attn = MultiHeadAttention(d_model, num_heads, rng=rng)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, rng=rng)
        self.ffn = FeedForward(d_model, d_ff, rng=rng)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def forward(
        self,
        x: np.ndarray,
        encoder_out: np.ndarray,
        tgt_mask: np.ndarray | None = None,
        src_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        attn_out, _ = self.self_attn.forward(x, mask=tgt_mask)
        x = self.norm1.forward(x + attn_out)
        cross_out, _ = self.cross_attn.forward(x, x_kv=encoder_out, mask=src_mask)
        x = self.norm2.forward(x + cross_out)
        ffn_out = self.ffn.forward(x)
        x = self.norm3.forward(x + ffn_out)
        return x


class Encoder:
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        seed: int = 0,
    ) -> None:
        self.layers = [
            EncoderLayer(d_model, num_heads, d_ff, seed=seed + i)
            for i in range(num_layers)
        ]

    def forward(self, x: np.ndarray, src_mask: np.ndarray | None = None) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, src_mask=src_mask)
        return x


class Decoder:
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        seed: int = 0,
    ) -> None:
        self.layers = [
            DecoderLayer(d_model, num_heads, d_ff, seed=seed + i)
            for i in range(num_layers)
        ]

    def forward(
        self,
        x: np.ndarray,
        encoder_out: np.ndarray,
        tgt_mask: np.ndarray | None = None,
        src_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, encoder_out, tgt_mask=tgt_mask, src_mask=src_mask)
        return x
