"""A minimal Transformer model implemented from scratch using NumPy."""

from __future__ import annotations

import numpy as np

from .attention import make_causal_mask
from .encoder_decoder import Encoder, Decoder


class TokenEmbedding:
    def __init__(self, vocab_size: int, d_model: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.emb = rng.normal(scale=1.0 / np.sqrt(d_model), size=(vocab_size, d_model))

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        return self.emb[token_ids]


class PositionalEncoding:
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        positions = np.arange(max_len)[:, None]
        dims = np.arange(d_model)[None, :]
        angle_rates = 1.0 / np.power(10000.0, (2 * (dims // 2)) / d_model)
        angles = positions * angle_rates

        pe = np.zeros((max_len, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(angles[:, 0::2])
        pe[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = pe

    def forward(self, x: np.ndarray) -> np.ndarray:
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]


class Linear:
    def __init__(self, in_dim: int, out_dim: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W = rng.normal(scale=1.0 / np.sqrt(in_dim), size=(in_dim, out_dim))
        self.b = np.zeros((out_dim,), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b


def create_padding_mask(token_ids: np.ndarray, pad_id: int = 0) -> np.ndarray:
    """Return boolean mask of shape (batch, 1, 1, seq_len)."""
    keep = token_ids != pad_id
    return keep[:, None, None, :]


def combine_masks(mask_a: np.ndarray | None, mask_b: np.ndarray | None) -> np.ndarray | None:
    if mask_a is None:
        return mask_b
    if mask_b is None:
        return mask_a
    return mask_a & mask_b


class Transformer:
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 512,
        seed: int = 0,
    ) -> None:
        self.d_model = d_model
        self.embedding = TokenEmbedding(vocab_size, d_model, seed=seed)
        self.positional = PositionalEncoding(d_model, max_len=max_len)
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, seed=seed + 1)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, seed=seed + 2)
        self.output_proj = Linear(d_model, vocab_size, seed=seed + 3)

    def forward(
        self,
        src_tokens: np.ndarray,
        tgt_tokens: np.ndarray,
        src_mask: np.ndarray | None = None,
        tgt_mask: np.ndarray | None = None,
        pad_id: int = 0,
    ) -> np.ndarray:
        """Compute logits for target tokens.

        Args:
            src_tokens: (batch, src_len) token ids.
            tgt_tokens: (batch, tgt_len) token ids.
            src_mask: optional boolean mask (batch, 1, 1, src_len).
            tgt_mask: optional boolean mask (batch, 1, tgt_len, tgt_len).
            pad_id: padding token id used for automatic masks if not provided.
        """
        if src_mask is None:
            src_mask = create_padding_mask(src_tokens, pad_id=pad_id)
        if tgt_mask is None:
            tgt_padding = create_padding_mask(tgt_tokens, pad_id=pad_id)
            tgt_causal = make_causal_mask(tgt_tokens.shape[1])
            tgt_mask = combine_masks(tgt_padding, tgt_causal)

        src_embed = self.positional.forward(self.embedding.forward(src_tokens))
        enc_out = self.encoder.forward(src_embed, src_mask=src_mask)

        tgt_embed = self.positional.forward(self.embedding.forward(tgt_tokens))
        dec_out = self.decoder.forward(tgt_embed, enc_out, tgt_mask=tgt_mask, src_mask=src_mask)

        logits = self.output_proj.forward(dec_out)
        return logits
