import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader
from net import Net
import torch.optim as optim

# 设置transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)

BATCH_SIZE = 16

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

net = Net().to('cuda')

criterion = nn.CrossEntropyLoss()  # 交叉式损失函数

optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)  # 优化器

# 定义函数
EPOCHS = 200

for epoch in range(EPOCHS):
    train_loss = 0.0
    for i, (datas, labels) in enumerate(train_loader):
        datas, labels = datas.to('cuda'), labels.to('cuda')
        # 梯度置零
        optimizer.zero_grad()
        # 训练
        outputs = net(datas)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
        # 累计损失
        train_loss += loss.item()
    # 打印信息
    print(epoch + 1, i + 1, train_loss / len(train_loader.dataset))

