import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # 卷积1 (3*32*32) => (32*32*32) => (32*16*16)
        x = self.pool(F.relu(self.conv1(x)))
        # 卷积2 (32*16*16) => (64*16*16) => (64*8*8)
        x = self.pool(F.relu(self.conv2(x)))
        # 卷积2 (64*8*8) => (128*8*8) => (128*4*4)
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 2048)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
