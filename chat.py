
import torch
from torch import nn

import config

def generate(model, context, gen_len):
    out = context
    for _ in range(gen_len):
        out = out[:, -config.window_size:]
        logits = model(out)[:, -1, :]
        probs = nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        out = torch.cat([out, next_token], dim=1)
    
    return out