import os

import torchinfo as torchinfo
from torch.optim.lr_scheduler import ReduceLROnPlateau

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader
from net import Net
import torch.optim as optim
import time
from net import BATCH_SIZE,learning_rate,EPOCHS

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=5, padding_mode='edge'),
    transforms.RandomRotation(45),
    transforms.ColorJitter(0.3, 0, 0, 0),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_set = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
net = Net().to(device)
# print(net)

criterion = nn.CrossEntropyLoss()  # 交叉式损失函数
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)  # 优化器
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.5, last_epoch=-1)  #学习率衰减
scheduler = ReduceLROnPlateau(optimizer, 'min')

all_train_loss = []
all_train_acc = []
all_valid_loss = []
all_valid_acc = []
all_learning_rate = []

for epoch in range(EPOCHS):
    since = time.time()
    train_loss = 0.0
    train_acc = 0.0
    net.train()

    for data, label in train_loader:
        data, label = data.to(device), label.to(device)
        # 梯度置零
        optimizer.zero_grad()
        # 训练
        outputs = net(data)
        # 计算损失
        # print(outputs.shape)
        # print(label.shape)
        loss = criterion(outputs, label)
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
        # 累计损失
        train_loss += loss.item()

        _, pred = outputs.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / label.shape[0]
        train_acc = train_acc + acc
    all_train_acc.append(train_acc / len(train_loader))
    all_train_loss.append(train_loss / len(train_loader))

    test_loss = 0
    eval_acc = 0
    net.eval()
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)
        output = net(data)
        # 记录单批次一次batch的loss，并且测试集不需要反向传播更新网络
        loss = criterion(output, label)
        test_loss = test_loss + loss.item()
        _, pred = output.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / label.shape[0]
        eval_acc = eval_acc + acc
    all_valid_acc.append(eval_acc / len(test_loader))
    all_valid_loss.append(test_loss / len(test_loader))
    all_learning_rate.append(optimizer.param_groups[0]['lr'])
    scheduler.step(test_loss)
    time_elapsed = time.time() - since
    print('epoch: {},train_loss: {:.4f},train_acc: {:.4f},eval_loss: {:.4f},eval_acc: {:.4f}'
          ' complete in {:.0f}m {:.0f}s'.format(epoch + 1, train_loss / len(train_loader), train_acc / len(train_loader)
                                                , test_loss / len(test_loader), eval_acc / len(test_loader),
                                                time_elapsed // 60, time_elapsed % 60))

# print(all_train_acc)
# print(all_valid_acc)
index = all_valid_acc.index(max(all_valid_acc))
print('best:train_loss: {:.4f},train_acc: {:.4f},eval_loss: {:.4f},eval_acc: {:.4f}'.format(all_train_loss[index],
                                                                                          all_train_acc[index],
                                                                                          all_valid_loss[index],
                                                                                          all_valid_acc[index]))
curr_time = int(time.time() * 1000)
plt.xlabel('EPOCHS')
plt.plot(all_train_acc, label="Train Accuracy")
plt.plot(all_valid_acc, color="red", label="Valid Accuracy")
plt.legend(loc='upper left')
plt.title('Accuracy\nEPOCHS: {}-BATCH_SIZE: {}-learning_rate: {}'.format(EPOCHS, BATCH_SIZE, learning_rate))
plt.savefig('./result/accuracy_{}.png'.format(curr_time))
plt.close()

plt.xlabel('EPOCHS')
plt.plot(all_train_loss, label="Train Loss")
plt.plot(all_valid_loss, color="red", label="Valid Loss")
plt.legend(loc='upper left')
plt.title('Loss\nEPOCHS: {}-BATCH_SIZE: {}-learning_rate: {}'.format(EPOCHS, BATCH_SIZE, learning_rate))
plt.savefig('./result/loss_{}.png'.format(curr_time))
plt.close()

plt.xlabel('EPOCHS')
plt.plot(all_learning_rate, label="Learning Rate")
plt.legend(loc='upper left')
plt.title('Learning Rate\nEPOCHS: {}-BATCH_SIZE: {}-learning_rate: {}'.format(EPOCHS, BATCH_SIZE, learning_rate))
plt.savefig('./result/lr_{}.png'.format(curr_time))
plt.close()
