
import torch
import wandb

from dataset import Dataset
from model import GPT
from chat import generate
import config


def train(train_dataloader, val_dataloader, model, optim, loss_fn):
    loop = train_dataloader

    total_loss, running_loss, running_count = 0, 0, 0
    for i, (x, y) in enumerate(loop):
        x, y = x.to(config.device), y.to(config.device)
        out = model(x)
        B, T, C = out.shape
        loss = loss_fn(out.view(B*T, C), y.view(B*T))
        total_loss += loss.item()
        running_loss += loss.item()
        running_count += 1

        optim.zero_grad()
        loss.backward()
        optim.step()

        if i > 0 and i % (len(train_dataloader) // 10) == 0:  # log 10 times per epoch
            train_loss = running_loss / running_count
            val_loss = eval(val_dataloader, model, loss_fn)
            log = {
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            wandb.log(log)

            running_loss, running_count = 0, 0

    return total_loss / len(train_dataloader)


def eval(dataloader, model, loss_fn):
    loop = dataloader
    model.eval()

    total_loss = 0
    for _, (x, y) in enumerate(loop):
        x, y = x.to(config.device), y.to(config.device)
        with torch.no_grad():
            out = model(x)
        B, T, C = out.shape
        total_loss += loss_fn(out.view(B*T, C), y.view(B*T)).item()

    model.train()
    return total_loss / len(dataloader)


def main():
    print(f'Save file: {config.save_model_file}')
    print(f'Epochs: {config.epochs}')
    print()

    train_dataset = Dataset(config.dataset_path + '/train.txt')
    val_dataset = Dataset(config.dataset_path + '/val.txt')
    test_dataset = Dataset(config.dataset_path + '/test.txt')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    print('Created datasets and dataloaders')

    model = GPT(config.vocab_size).to(config.device)
    optim = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    wandb.init(project='me-gpt', entity='willjhliang')
    wandb.config = {
        'epochs': config.epochs,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'window_size': config.window_size,
        'embed_size': config.embed_size,
        'num_heads': config.num_heads,
        'num_blocks': config.num_blocks,
        'dropout': config.dropout,
    }

    for epoch in range(config.epochs):
        print(f'Epoch: {epoch}')
        train_loss = train(train_dataloader, val_dataloader, model, optim, loss_fn)
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
    print(generate(model, torch.tensor([[config.stoi['\n']]]).to(config.device), 100).tolist()[0])


if __name__ == '__main__':
    main()