import torch
import torch.nn as nn
from LeNet import LeNet
import torchvision as tv
import torchvision.transforms as transforms
import copy
import torch.optim as optim
import argparse
import platform
import math
import time
import numpy as np
import collections

class PublicKey:
    def __init__(self, A, P, n, s):
        self.A = A
        self.P = P
        self.n = n
        self.s = s

    def __repr__(self):
        return 'PublicKey({}, {}, {}, {})'.format(self.A, self.P, self.n, self.s)

from cuda_test import KeyGen, Enc, Dec

# 超参数设置
EPOCH_NUM = 15  # 遍历数据集次数
BATCH_SIZE = 64  # 批处理尺寸(batch_size)
LR = 0.001  # 学习率
CLIENT_NUM = 2 # client 数目

# 同态加密设置
prec = 32
bound = 2 ** 3

device = torch.device("cuda")
torch.cuda.set_device(2)

# 加载数据集
transform = transforms.ToTensor()  # 定义数据预处理方式

MNIST_data = "/home/dchen/dataset"  # linux

# 定义训练数据集
trainset = tv.datasets.MNIST(
    root=MNIST_data,
    train=True,
    download=False,
    transform=transform)

# 定义训练批处理数据
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    )

# 定义测试数据集
testset = tv.datasets.MNIST(
    root=MNIST_data,
    train=False,
    download=False,
    transform=transform)

# 定义测试批处理数据
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    )

# 分割训练集
dataset_list = list(trainloader)
dataset_len = len(dataset_list)
client_len = dataset_len // CLIENT_NUM

# 密钥生成
pk, sk = KeyGen()

# 网络参数初始化
def weight_init(m):
    # 使用isinstance来判断m属于什么类型
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # torch.manual_seed(7)   # 随机种子，是否每次做相同初始化赋值
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        # m中的 weight，bias 其实都是 Variable，为了能学习参数以及后向传播
        m.weight.data.fill_(1)
        m.bias.data.zero_()

net = LeNet()
# 初始化网络参数
net.apply(weight_init)  # apply函数会递归地搜索网络内的所有module并把参数表示的函数应用到所有的module上

# 定义损失函数
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
optimizer_server = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

# 分配用户参数
client_0_net = LeNet().to(device)
client_1_net = LeNet().to(device)

model_parameters = net.state_dict()  # 提取网络参数

model_parameters_dict = collections.OrderedDict() 
for key, value in model_parameters.items():
    model_parameters_dict[key] = torch.numel(value), value.shape

# client训练，获取梯度并加密
def get_client_encrypted_grad(client_inputs, client_labels, net_dict, client_net):
    client_net.load_state_dict(net_dict)
    client_outputs = client_net(client_inputs)
    client_loss = criterion(client_outputs, client_labels)
    client_optimizer = optim.SGD(client_net.parameters(), lr=LR, momentum=0.9)
    client_optimizer.zero_grad()  # 梯度置零 
    client_loss.backward()  # 求取梯度

    params_modules = list(client_net.named_parameters())
    params_grad_list = []
    for params_module in params_modules:
        name, params = params_module
        params_grad_list.append(copy.deepcopy(params.grad).view(-1))

    params_grad = ((torch.cat(params_grad_list, 0) + bound) * 2 ** prec).long().cuda()
    client_encrypted_grad = Enc(pk, params_grad)

    client_optimizer.zero_grad()  # 梯度置零

    return client_encrypted_grad

st = time.time()

for epoch in range(EPOCH_NUM):
    # 处理数据
    for index in range(client_len):
        net_dict = net.state_dict()  # 提取网络参数
        
        # client 0
        client_0_inputs, client_0_labels = dataset_list[index]
        client_0_inputs, client_0_labels = client_0_inputs.to(device), client_0_labels.to(device)
        client_0_encrypted_grad = get_client_encrypted_grad(client_0_inputs, client_0_labels, net_dict, client_0_net)
        # client 1
        client_1_inputs, client_1_labels = dataset_list[index + client_len]
        client_1_inputs, client_1_labels = client_1_inputs.to(device), client_1_labels.to(device)
        client_1_encrypted_grad = get_client_encrypted_grad(client_1_inputs, client_1_labels, net_dict, client_1_net)

        # 计算新的梯度
        encrypted_sum = client_0_encrypted_grad + client_1_encrypted_grad
        data_sum = Dec(sk, encrypted_sum).float() / (2 ** prec) / CLIENT_NUM - bound

        ind = 0
        client_grad_dict = dict()
        for key in model_parameters_dict:
            params_size, params_shape = model_parameters_dict[key]
            client_grad_dict[key] = data_sum[ind : ind + params_size].reshape(params_shape)
            ind += params_size

        # 加载新的模型参数
        params_modules_server = net.to(device).named_parameters()
        for params_module in params_modules_server:
            name, params = params_module
            params.grad = client_grad_dict[name]  # 用字典中存储的子模型的梯度覆盖网络中的参数梯度
        optimizer_server.step() # 更新所有参数
    
    # 每跑完一次epoch测试一下准确率
    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            # 取得分最高的那个类
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('第%d个epoch的识别准确率为：%.2f%%' % (epoch + 1, (100 * float(correct) / total)))

print("Train Time: %.1f s/epoch" % ((time.time() - st) / EPOCH_NUM))

# 最终测试一下准确率
with torch.no_grad():
    correct = 0
    total = 0 
    for i, data in enumerate(testloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
print('------------------------------------------------------')
print('最终识别准确率为：%.2f%% (%d / %d)' % ((100 * float(correct) / total), correct, total))
print('------------------------------------------------------')

torch.save(net.state_dict(), 'models/demo_%d.pth' % (epoch + 1))
print('successfully save the model to models/demo_%d.pth' % (epoch + 1))


