import torch.nn as nn
import torch.nn.functional as F

EPOCHS = 100
learning_rate = 0.01
BATCH_SIZE = 256

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # conv1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.batch_normal1 = nn.BatchNorm2d(32)
        self.conv1_1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.batch_normal1_1 = nn.BatchNorm2d(32)
        # conv2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.batch_normal2 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.batch_normal2_1 = nn.BatchNorm2d(64)
        # conv3
        self.conv3 = nn.Conv2d(64, 164, kernel_size=3, padding=1)
        self.batch_normal3 = nn.BatchNorm2d(164)
        # conv4
        self.conv4 = nn.Conv2d(164, 328, kernel_size=3, padding=1)
        self.batch_normal4 = nn.BatchNorm2d(328)
        # conv5
        self.conv5 = nn.Conv2d(328, 256, kernel_size=3, padding=1)
        self.batch_normal5 = nn.BatchNorm2d(256)
        self.conv5_1 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.batch_normal5_1 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256*4*4, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)
        # self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # 卷积1 (3*32*32) => (32*32*32) => (32*16*16)
        # 卷积1 (3*32*32) => (32*32*32)
        x = F.relu(self.batch_normal1(self.conv1(x)))
        x = F.relu(self.batch_normal1_1(self.conv1_1(x)))
        # 卷积2 (32*16*16) => (64*16*16) => (64*8*8)
        # 卷积2 (32*32*32) => (64*32*32)
        x = F.relu(self.batch_normal2(self.conv2(x)))
        x = F.relu(self.batch_normal2_1(self.conv2_1(x)))
        # 卷积3 (64*32*32) => (128*32*32)
        x = F.relu(self.batch_normal3(self.conv3(x)))
        x = F.relu(self.batch_normal4(self.conv4(x)))
        x = F.relu(self.batch_normal5(self.conv5(x)))
        x = F.relu(self.batch_normal5_1(self.conv5_1(x)))
        x = x.view(-1, 256*4*4)
        nn.Dropout(0.75)
        x = F.relu(self.fc1(x))
        nn.Dropout(0.75)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
