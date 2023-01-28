
import torch
from tqdm import tqdm

from dataset import Dataset
from model import GPT
from chat import generate
import config


def train(dataloader, model, optim, loss_fn):
    loop = tqdm(dataloader)

    total_loss = 0
    for _, (x, y) in enumerate(loop):
        x, y = x.to(config.device), y.to(config.device)
        out = model(x)
        B, T, C = out.shape
        loss = loss_fn(out.view(B*T, C), y.view(B*T))
        total_loss += loss

        optim.zero_grad()
        loss.backward()
        optim.step()
    
    return total_loss / len(dataloader)


def eval(dataloader, model, loss_fn):
    loop = tqdm(dataloader)
    model.eval()

    total_loss = 0
    for _, (x, y) in enumerate(loop):
        x, y = x.to(config.device), y.to(config.device)
        out = model(x)
        B, T, C = out.shape
        total_loss += loss_fn(out.view(B*T, C), y.view(B*T))

    model.train()
    return total_loss / len(dataloader)


def main():
    print(f'Save file: {config.save_model_file}')
    print(f'Epochs: {config.epochs}')
    print()

    train_dataset = Dataset('dataset/train_small.txt')
    val_dataset = Dataset('dataset/val.txt')
    test_dataset = Dataset('dataset/test.txt')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    print('Created datasets and dataloaders')

    model = GPT(train_dataset.vocab_size).to(config.device)
    optim = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        print(f'Epoch: {epoch}')
        train_loss = train(train_dataloader, model, optim, loss_fn)
        val_loss = eval(val_dataloader, model, loss_fn)
        print(f'Train loss: {train_loss}')
        print(f'Validation loss: {val_loss}')
        print()
    
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict(),
    }
    torch.save(state, config.save_model_file)

    test_loss = eval(test_dataloader, model, loss_fn)
    print(f'Test loss: {test_loss}')

    print('Generated text:')
    print(generate(model, torch.tensor([[train_dataset.stoi['\n']]]).to(config.device), 100).tolist()[0])


if __name__ == '__main__':
    main()