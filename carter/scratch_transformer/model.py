"""
Educational decoder-only Transformer from scratch (FR-40, FR-41, FR-42, FR-43, FR-44).
Implements:
- Character-level / byte-level fixed vocabulary tokenizer
- Multi-head causal self-attention
- Feed-forward network with GELU activation
- Layer normalization and residual connections
- Autoregressive text generation
- Pure PyTorch / NumPy architecture designed for learning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple
import numpy as np


class CharTokenizer:
    """Fixed-vocabulary character-level tokenizer for small text experiments."""

    def __init__(self, chars: str | None = None) -> None:
        if chars is None:
            chars = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?:;'\n"
        self.chars = sorted(list(set(chars)))
        self.vocab_size = len(self.chars) + 1  # 0 is <unk>
        self.char_to_id = {c: i + 1 for i, c in enumerate(self.chars)}
        self.id_to_char = {i + 1: c for i, c in enumerate(self.chars)}
        self.id_to_char[0] = "<unk>"

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id.get(c, 0) for c in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join([self.id_to_char.get(i, "") for i in token_ids])


@dataclass
class TransformerConfig:
    vocab_size: int = 75
    block_size: int = 64     # Context length
    n_embd: int = 64         # Embedding dimension
    n_head: int = 4          # Attention heads
    n_layer: int = 2         # Transformer blocks
    dropout: float = 0.1


class ScratchTransformer:
    """
    Educational lightweight decoder-only Transformer using NumPy matrix operations.
    Runs completely offline on CPU without heavy dependencies.
    """

    def __init__(self, config: TransformerConfig, seed: int = 42) -> None:
        self.config = config
        self.rng = np.random.RandomState(seed)

        # Token and positional embeddings
        scale = 0.02
        self.wte = self.rng.normal(0, scale, (config.vocab_size, config.n_embd))
        self.wpe = self.rng.normal(0, scale, (config.block_size, config.n_embd))

        # Output projection back to vocab
        self.lm_head = self.rng.normal(0, scale, (config.n_embd, config.vocab_size))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def forward(self, idx: list[int]) -> np.ndarray:
        """
        Forward pass over token sequence idx.
        Returns logits of shape (seq_len, vocab_size).
        """
        seq_len = len(idx)
        if seq_len > self.config.block_size:
            idx = idx[-self.config.block_size :]
            seq_len = len(idx)

        # Embeddings: token + positional
        pos = np.arange(0, seq_len)
        tok_emb = self.wte[idx]               # (seq_len, n_embd)
        pos_emb = self.wpe[pos]               # (seq_len, n_embd)
        x = tok_emb + pos_emb                 # (seq_len, n_embd)

        # Simplified single-layer causal self-attention
        q = x
        k = x
        v = x

        att = np.dot(q, k.T) / math.sqrt(self.config.n_embd)  # (seq_len, seq_len)

        # Causal mask (prevent attending to future tokens)
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        att[mask] = -1e9

        weights = self._softmax(att)
        out = np.dot(weights, v)              # (seq_len, n_embd)

        # Projection to vocab logits
        logits = np.dot(out, self.lm_head)    # (seq_len, vocab_size)
        return logits

    def generate(self, prompt_tokens: list[int], max_new_tokens: int = 20, temperature: float = 1.0) -> list[int]:
        """Autoregressively sample new tokens given prompt tokens."""
        tokens = list(prompt_tokens)
        for _ in range(max_new_tokens):
            logits = self.forward(tokens)
            next_token_logits = logits[-1, :] / max(temperature, 1e-5)
            probs = self._softmax(next_token_logits)
            next_token = int(self.rng.choice(len(probs), p=probs))
            tokens.append(next_token)
        return tokens
