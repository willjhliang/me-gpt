
import torch
from torch import nn

from model import GPT
from utils import encode, decode
import config


def generate(model, context, gen_len):
    out = context
    for _ in range(gen_len):
        out = out[:, -config.window_size:]
        logits = model(out)[:, -1, :]
        probs = nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        out = torch.cat([out, next_token], dim=1)
        print(decode(out[0].tolist()[-1]), end='', flush=True)
    
    return out


def main():
    model = GPT(config.vocab_size).to(config.device)
    checkpoint = torch.load(config.load_model_file, map_location=config.device)
    model.load_state_dict(checkpoint['state_dict'])

    prompt = input('Enter prompt: ')
    context = '\nOTHER\n' + prompt + '\n\nWILL\n'
    print(context, end='', flush=True)
    generate(model, torch.tensor([encode(context)]).to(config.device), 1000)


if __name__ == '__main__':
    main()