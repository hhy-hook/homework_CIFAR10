import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader
from net import Net
import torch.optim as optim
import time

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_set = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)


#定义hyper-parameter
EPOCHS = 50
BATCH_SIZE = 32
learning_rate = 0.01

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
net = Net().to(device)

criterion = nn.CrossEntropyLoss()  # 交叉式损失函数
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)  # 优化器
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.5, last_epoch=-1)  #学习率衰减


all_train_acc = []
all_valid_acc = []

for epoch in range(EPOCHS):
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
        loss = criterion(outputs, label)
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
        scheduler.step()
        # 累计损失
        train_loss += loss.item()

        _, pred = outputs.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / label.shape[0]
        train_acc = train_acc + acc
    all_train_acc.append(train_acc / len(train_loader))
    print('epoch: {}, train_loss: {:.4f},train_acc: {:.4f}'.format(epoch + 1, train_loss / len(train_loader),
                                                                   train_acc / len(train_loader)))

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
    print('epoch: {}, evalloss: {:.4f},evalacc: {:.4f}'.format(epoch + 1, test_loss / len(test_loader),
                                                               eval_acc / len(test_loader)))

print(all_train_acc)
print(all_valid_acc)
plt.plot(all_train_acc, label="Train Accuracy")
plt.plot(all_valid_acc, color="red", label="Valid Accuracy")
plt.legend(loc='upper left')
plt.title('EPOCHS: {}-BATCH_SIZE: {}-learning_rate: {}'.format(EPOCHS,BATCH_SIZE,learning_rate))
plt.savefig('./result/{}.png'.format(int(time.time() * 1000)))