
# Learning parameters
epochs = 3
batch_size = 256
learning_rate = 1e-3

# Run configuration
save_model_file = 'model.pth.tar'
load_model_file = 'model.pth.tar'

# Model configuration
window_size = 256
embed_size = 768
num_heads = 12
num_blocks = 6
dropout = 0.5

# Data globals
dataset_path = '../input/messenger-texts'
with open(dataset_path + '/train.txt', 'r') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Misc
device = 'cuda'
