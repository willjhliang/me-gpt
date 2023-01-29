
import config


def encode(text):
    return [config.stoi[c] for c in text]


def decode(tokens):
    if type(tokens) == int:
        return config.itos[tokens]
    return ''.join([config.itos[i] for i in tokens])