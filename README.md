# homework_CIFAR10
## Introduction

## Installation
### Step 1: Clone the Code from Github
```bash
git clone https://github.com/hhy-hook/homework_CIFAR10.git
cd homework_CIFAR10/
```
### Step 2: Install Requirements
Python: see requirement.txt for complete list of used packages. We recommend doing a clean installation of requirements using virtualenv:
```bash
conda create -n envname python=3.7
conda activate envname
pip install -r requirement.txt 
```
## Running Task
```bash
python ./main.py
```
## Structs
```
homework
|
| img.png
|
| main.py #主要运行程序
|
| net.py #学习网络的结构
|
└───dataset #下载数据集(运行下载后，结构如下)
|   |
|   └─── cifar-10-batches-py
|   └─── cifar-10-python.tar.gz
```
## Usage
- win11 + anaconda3 + python 3.7 + cuda11.3 + torch 11.0
- torch资源包：https://download.pytorch.org/whl/torch_stable.html 选择![img.png](img.png)
- torchvison安装：pip install torchvision==0.12.0
### 1.torch资源包安装
```bash
$ conda activate YOUR_CONDA_ENV(homework_CIFAR10)
$ cd whl目录
$ pip install torch-1.11.0+cu113-cp37-cp37m-win_amd64.whl
```
### 2.如果出现torch和torchvision版本不匹配问题
```bash
# 通过pycharm删除torchvision单个库
$ pip install torchvision==0.12.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
