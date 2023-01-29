
# Learning parameters
epochs = 4
batch_size = 128
learning_rate = 3e-4

# Run configuration
save_model_file = 'model.pth.tar'
load_model_file = 'model.pth.tar'

# Model configuration
window_size = 256
embed_size = 384
num_heads = 6
num_blocks = 6
dropout = 0.3

# Data globals
with open('../input/messenger-texts/train.txt', 'r') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Misc
device = 'cuda'
