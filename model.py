
import torch
from torch import nn

import config


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(config.embed_size, head_size, bias=False)
        self.key = nn.Linear(config.embed_size, head_size, bias=False)
        self.value = nn.Linear(config.embed_size, head_size, bias=False)
        self.register_buffer('t', torch.tril(torch.ones(config.window_size, config.window_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)
        e = q @ k.transpose(-2, -1) / (C ** 0.5)
        e = e.masked_fill(self.t[:T, :T] == 0, float('-inf'))
        a = nn.functional.softmax(e, dim=-1)
        return a @ v
        

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, config.embed_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class TransformerBlock(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        head_size = config.embed_size // num_heads
        self.self_attention = MultiHeadAttention(num_heads, head_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.embed_size, 4 * config.embed_size),
            nn.ReLU(),
            nn.Linear(4 * config.embed_size, config.embed_size),
            nn.Dropout(config.dropout)
        )
        self.ln1, self.ln2 = nn.LayerNorm(config.embed_size), nn.LayerNorm(config.embed_size)
    
    def forward(self, x):
        x = x + self.self_attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, config.embed_size)
        self.position_embedding = nn.Embedding(config.window_size, config.embed_size)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config.num_heads) for _ in range(config.num_blocks)]
        )
        self.ln = nn.LayerNorm(config.embed_size)
        self.linear = nn.Linear(config.embed_size, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.token_embedding(x) + self.position_embedding(torch.arange(T, device=config.device))
        x = self.transformer_blocks(x)
        x = self.linear(self.ln(x))
        return x

    def generate(self):
        pass