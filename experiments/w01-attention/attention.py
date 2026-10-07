"""Readable CPU decoder: causal attention, GQA and a per-layer KV cache.

This is a teaching model, not a serving implementation. Cache concatenation and
KV head expansion intentionally favor readability over memory efficiency.
"""
import math
import torch
from torch import nn


def rope(x, positions):
    """Rotate adjacent feature pairs; x has shape [B, H, T, d]."""
    d = x.shape[-1]
    frequencies = 10000.0 ** (-torch.arange(0, d, 2, dtype=x.dtype) / d)
    angles = positions.to(x.dtype)[:, None] * frequencies[None, :]
    cosine, sine = angles.cos()[None, None], angles.sin()[None, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cosine - odd * sine,
                        even * sine + odd * cosine), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, dim=32, query_heads=4, kv_heads=2):
        super().__init__()
        assert dim % query_heads == 0 and query_heads % kv_heads == 0
        self.hq, self.hkv, self.d = query_heads, kv_heads, dim // query_heads
        assert self.d % 2 == 0
        self.q = nn.Linear(dim, self.hq * self.d, bias=False)
        self.k = nn.Linear(dim, self.hkv * self.d, bias=False)
        self.v = nn.Linear(dim, self.hkv * self.d, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cache=None, *, wrong_mask=False, reset_positions=False):
        batch, length, _ = x.shape
        past = 0 if cache is None else cache[0].shape[2]
        def split(projection, heads):
            return projection(x).view(batch, length, heads, self.d).transpose(1, 2)
        positions = torch.arange(length) + (0 if reset_positions else past)
        q = rope(split(self.q, self.hq), positions)
        k = rope(split(self.k, self.hkv), positions)
        v = split(self.v, self.hkv)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        new_cache = (k, v)  # Store compact Hkv heads, not the expanded Hq heads.
        group_size = self.hq // self.hkv
        k_expanded = k.repeat_interleave(group_size, dim=1)
        v_expanded = v.repeat_interleave(group_size, dim=1)
        scores = q @ k_expanded.transpose(-1, -2) / math.sqrt(self.d)
        query_indices = torch.arange(length) + (0 if wrong_mask else past)
        key_indices = torch.arange(k.shape[2])
        allowed = key_indices[None, :] <= query_indices[:, None]
        weights = scores.masked_fill(~allowed, -torch.inf).softmax(dim=-1)
        y = (weights @ v_expanded).transpose(1, 2).contiguous().view(batch, length, -1)
        return self.out(y), new_cache


class Block(nn.Module):
    def __init__(self, dim, hq, hkv):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = Attention(dim, hq, hkv)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x, cache=None, **options):
        a, cache = self.attention(self.norm1(x), cache, **options)
        x = x + a
        return x + self.ffn(self.norm2(x)), cache


class TinyDecoder(nn.Module):
    def __init__(self, kv_heads=2, vocab=41, dim=32, layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([Block(dim, 4, kv_heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, token_ids, caches=None, **options):
        x = self.embedding(token_ids)
        new_caches = []
        for index, block in enumerate(self.blocks):
            cache = None if caches is None else caches[index]
            x, new_cache = block(x, cache, **options)
            new_caches.append(new_cache)
        return self.lm_head(self.norm(x)), new_caches
