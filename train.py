import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F
from load_dataset import get_data_loaders
import torch.optim as optim
import json

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # h, w = floor(((h + 2* padding - dialation * (k-1)-1)/stride)+1)
        # current input is 32X32 -> [B, 3, 32, 32]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, padding=1, kernel_size=4) # [B, 64, 31, 31]
        self.relu1 = nn.ReLU()
        self.b_norm1 = nn.BatchNorm2d(64)
        self.dp1 = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3) #[B, 128, 29, 29]
        self.relu2 = nn.ReLU()
        self.b_norm2 = nn.BatchNorm2d(128)
        self.pooling1 = nn.MaxPool2d(kernel_size=3, stride=3) # [B, 128, 9, 9]
        self.dp2 = nn.Dropout(0.1)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1) # [B, 128*9*9]
        self.hidden1 = nn.Linear(in_features=128*9*9, out_features=500)
        self.proj = nn.Linear(in_features=500, out_features=10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.b_norm1(x)
        x = self.dp1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.b_norm2(x)
        x = self.pooling1(x)
        x = self.dp2(x)
        x = self.flatten(x)
        x = self.hidden1(x)
        out = self.proj(x)
        return out



def validation_loss(model, target_data):
    target_data = val_dataloader
    with torch.no_grad():
        model.eval()
        lossi = []
        lSoftmax = nn.LogSoftmax(dim=-1)
        b_count = 0
        accuracy = 0
        for xb, yb in target_data:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            probs = lSoftmax(out)
            loss = F.cross_entropy(out, yb.view(-1))
            indices = torch.argmax(probs, dim=-1)
            accuracy += (indices == yb).sum().item()
            b_count += yb.size(0)
            lossi.append(loss.item())
            xb = xb.to('cpu')
            yb = yb.to('cpu')
        lossi = torch.tensor(lossi).mean()
        model.train()
    return {
        "loss": lossi.item(),
        "accuracy": accuracy/b_count
    }



if __name__ == '__main__':
    train_dataloader, val_dataloader = get_data_loaders(download=True)
    print(f"Data is loaded:- ")
    # import pdb; pdb.set_trace()
    model = CNNModel()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr = 1e-3)
    lossi = []
    stepi = []
    vallossi = []
    model.train()
    step = 0
    for epoch in range(6):
        for xb, yb in train_dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb.view(-1))
            loss.backward()
            optimizer.step()
            xb = xb.to('cpu')
            yb = yb.to('cpu')
            if step%100 == 0:
                val_stats = validation_loss(model, val_dataloader)
                print(f"Epoch:- {epoch}, step:- {step}, train_loss:- {loss.item()}, val_loss:- {val_stats['loss']}")
                lossi.append(loss.item())
                stepi.append(step)
                vallossi.append(val_stats['loss'])
            step+=1
    
    plt.figure(figsize=(10, 6))
    plt.plot(stepi, lossi, label='Training Loss', color='blue')
    plt.plot(stepi, vallossi, label='Validation Loss', color='orange')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_plot.png')
    print("Plot saved as training_loss_plot.png")
    plt.show()

    test_dataloader, _ = get_data_loaders(download=True, split='valid')
    stats = validation_loss(model, test_dataloader)
    with open('test_stats.json', 'w') as file:
        json.dump(stats, file, indent=4)



    






