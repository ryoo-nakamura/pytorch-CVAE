import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

#パラメータ
CLASS_SIZE = 10
BATCH_SIZE = 256
ZDIM = 2
NUM_EPOCHS = 200

#GPUの設定
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#学習モデルの構築
class CVAE(nn.Module):
    def __init__(self, zdim):
        super().__init__()
        self._zdim = zdim #latent dimantion
        self._in_units = 28 * 28
        hidden_units = 512
        self._to_mean = nn.Linear(hidden_units, zdim)
        self._to_lnvar = nn.Linear(hidden_units, zdim)
        
        # Encoder Sequential 
        self._encoder = nn.Sequential(
            nn.Linear(self._in_units + CLASS_SIZE, hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(inplace=True),
        )
        # Decorder Sequential
        self._decoder = nn.Sequential(
            nn.Linear(zdim + CLASS_SIZE, hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units, self._in_units),
            nn.Sigmoid()
        )

    def encode(self, x, labels):
        in_ = torch.empty((x.shape[0], self._in_units + CLASS_SIZE), device=device)#(バッチサイズ,隠れ層のノード + one-hot Label)
        in_[:, :self._in_units] = x #_in_units数より後ろがxの値
        in_[:, self._in_units:] = labels #_in_units数より前labelsの値
        h = self._encoder(in_)
        mean = self._to_mean(h)
        lnvar = self._to_lnvar(h)
        return mean, lnvar

    def decode(self, z, labels):
        in_ = torch.empty((z.shape[0], self._zdim + CLASS_SIZE), device=device)
        in_[:, :self._zdim] = z
        in_[:, self._zdim:] = labels
        return self._decoder(in_)

#one-hotベクトル変換用関数
def to_onehot(label):
    return torch.eye(CLASS_SIZE, device=device, dtype=torch.float32)[label]

# 学習データの呼び込み
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

#モデルの呼び出しと最適化関数の設定
#GPUがある場合はGPUを利用
model = CVAE(ZDIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train()
for e in range(NUM_EPOCHS):
    train_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        labels = to_onehot(labels)
        # Reconstruction images
        
        # Encode images
        x = images.view(-1, 28*28*1).to(device)
        mean, lnvar = model.encode(x, labels)
        std = lnvar.exp().sqrt()
        epsilon = torch.randn(ZDIM, device=device)

        # Decode latent variables
        z = mean + std * epsilon
        y = model.decode(z, labels)

        # Compute loss
        kld = 0.5 * (1 + lnvar - mean.pow(2) - lnvar.exp()).sum(axis=1) #pow(a,2)２畳の計算
        bce = F.binary_cross_entropy(y, x, reduction='none').sum(axis=1)
        loss = (-1 * kld + bce).mean()

        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.shape[0]

    print(f'epoch: {e + 1} epoch_loss: {train_loss/len(train_dataset)}')



# Generation data with label 
for n in range(0,10):
    NUM_GENERATION = 100
    os.makedirs(f'/Users/nakamura/Downloads/CVAE/', exist_ok=True)
    os.makedirs(f'/Users/nakamura/Downloads/CVAE/label{n}/', exist_ok=True)
    model.eval()
    
    for i in range(NUM_GENERATION):
        z = torch.randn(ZDIM, device=device).unsqueeze(dim=0)
        label = torch.tensor([n], device=device)
        with torch.no_grad():
            y = model.decode(z, to_onehot(label))
        y = y.reshape(28, 28).cpu().detach().numpy()
    
        # Save image
        fig, ax = plt.subplots()
        ax.imshow(y)
        ax.set_title(f'Generation(label={label.cpu().detach().numpy()[0]})')
        ax.tick_params(
            labelbottom=False,
            labelleft=False,
            bottom=False,
            left=False,
        )
        plt.savefig(f'/Users/nakamura/Downloads/CVAE/label{n}/img{i + 1}')
        plt.close(fig) 