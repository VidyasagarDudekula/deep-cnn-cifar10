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


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, downsample=None, residual=True):
        super().__init__()
        # [B, 3, 32, 32]
        # h, w = floor(((h + p*2 -(k-1)-1)/stride)+1)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1) #[B, 64, 32, 32]
        self.relu1 = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes*2, kernel_size=3, stride=1, padding=1) #[B, 128, 32, 32]
        self.relu2 = nn.ReLU()
        self.batch_norm2 = nn.BatchNorm2d(planes*2)
        self.conv3 = nn.Conv2d(planes*2, planes*4, kernel_size=3, stride=2) #[B, 256, 15, 15]
        self.relu3 = nn.ReLU()
        self.batch_norm3 = nn.BatchNorm2d(planes*4)
        
        self.downsample = downsample
        
        self.residual = residual
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)
        
        if self.residual:
            if self.downsample is not None:
                identity = self.downsample(identity)
                x = x + identity
        return x

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # h, w = floor(((h + 2* padding - dialation * (k-1)-1)/stride)+1)
        # current input is 32X32 -> [B, 3, 32, 32]
        downsampling = nn.Sequential(
            nn.Conv2d(3, 4*64, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(256)
        )
        self.residual_block = ResidualBlock(3, 64, downsampling) #[B, 256, 15, 15]
        self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=3) # [B, 256, 5, 5]
        self.dp2 = nn.Dropout(0.1)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1) # [B, 128*4*4]
        self.hidden1 = nn.Linear(in_features=256*5*5, out_features=100)
        self.layer_norm1 = nn.LayerNorm(100)
        self.relu1 = nn.ReLU()
        self.dp3 = nn.Dropout(0.1)
        self.hidden2 = nn.Linear(in_features=100, out_features=50)
        self.layer_norm2 = nn.LayerNorm(50)
        self.relu2 = nn.ReLU()
        self.dp4 = nn.Dropout(0.1)
        self.proj = nn.Linear(in_features=50, out_features=10)
    
    def forward(self, x):
        x = self.residual_block(x)
        x = self.pooling1(x)
        x = self.dp2(x)
        x = self.flatten(x)
        x = self.hidden1(x)
        x = self.layer_norm1(x)
        x = self.relu1(x)
        x = self.dp3(x)
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.layer_norm2(x)
        x = self.dp4(x)
        out = self.proj(x)
        return out



def validation_loss(model, target_data):
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
            accuracy += (indices == yb.view(-1)).sum().item()
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
    train_dataloader, val_dataloader = get_data_loaders(download=True, split='train', batch=32)
    print(f"Data is loaded:- ")
    # import pdb; pdb.set_trace()
    model = CNNModel()
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr = 1e-4)
    lossi = []
    stepi = []
    vallossi = []
    model.train()
    step = 0
    for epoch in range(15):
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
            if step%200 == 0:
                val_stats = validation_loss(model, val_dataloader)
                print(f"Epoch:- {epoch}, step:- {step}, train_loss:- {loss.item()}, val_loss:- {val_stats['loss']}, val_accuracy:- {val_stats['accuracy']}")
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
    # plt.show()

    test_dataloader, _ = get_data_loaders(download=True, split='valid')
    stats = validation_loss(model, test_dataloader)
    with open('test_stats.json', 'w') as file:
        json.dump(stats, file, indent=4)



    






