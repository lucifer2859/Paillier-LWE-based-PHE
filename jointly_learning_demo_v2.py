import torch
import torch.nn as nn
from LeNet import LeNet
from torchvision import datasets, transforms
import copy
import torch.optim as optim
import argparse
import platform
import math
import time

EPOCH_NUM = 100
BATCH_SIZE = 64
LR = 0.001
CLIENT_NUM = 2

device = torch.device("cuda")
torch.cuda.set_device(6)

transform = transforms.ToTensor()

trainset0 = datasets.ImageFolder('/home/dchen/dataset/MNIST/IID/train/client0/', transform=transform)
trainloader0 = torch.utils.data.DataLoader(
    trainset0,
    batch_size=BATCH_SIZE,
    shuffle=True
)

testset0 = datasets.ImageFolder('/home/dchen/dataset/MNIST/IID/test/client0/', transform=transform)
testloader0 = torch.utils.data.DataLoader(
    testset0,
    batch_size=BATCH_SIZE,
    shuffle=False
)

trainset1 = datasets.ImageFolder('/home/dchen/dataset/MNIST/IID/train/client1/', transform=transform)
trainloader1 = torch.utils.data.DataLoader(
    trainset1,
    batch_size=BATCH_SIZE,
    shuffle=True
)

testset1 = datasets.ImageFolder('/home/dchen/dataset/MNIST/IID/test/client1/', transform=transform)
testloader1 = torch.utils.data.DataLoader(
    testset1,
    batch_size=BATCH_SIZE,
    shuffle=False
)

dataset_list = list(trainloader0)
dataset_len = len(dataset_list)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

net = LeNet()
net.apply(weight_init)

criterion = nn.CrossEntropyLoss()
optimizer_server = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

client_0_net = LeNet().to(device)
client_1_net = LeNet().to(device)

def get_client_grad(client_inputs, client_labels, net_dict, client_net):
    client_net.load_state_dict(net_dict)
    client_outputs = client_net(client_inputs)
    client_loss = criterion(client_outputs, client_labels)
    client_optimizer = optim.SGD(client_net.parameters(), lr=LR, momentum=0.9)
    client_optimizer.zero_grad() 
    client_loss.backward()

    client_grad_dict = dict()
    params_modules = list(client_net.named_parameters())
    for params_module in params_modules:
        name, params = params_module
        params_grad = copy.deepcopy(params.grad)
        client_grad_dict[name] = params_grad
    client_optimizer.zero_grad()
    return client_grad_dict    

st = time.time()

for epoch in range(EPOCH_NUM):
    data_iter0 = iter(trainloader0)
    data_iter1 = iter(trainloader1)
    
    for index in range(dataset_len):
        net_dict = net.state_dict()

        client_0_inputs, client_0_labels = next(data_iter0)
        client_0_inputs = torch.index_select(client_0_inputs, 1, torch.LongTensor([0]))
        client_0_inputs, client_0_labels = client_0_inputs.to(device), client_0_labels.to(device)
        client_0_grad_dict = get_client_grad(client_0_inputs, client_0_labels, net_dict, client_0_net)
  
        client_1_inputs, client_1_labels = next(data_iter1)
        client_1_inputs = torch.index_select(client_1_inputs, 1, torch.LongTensor([0]))
        client_1_inputs, client_1_labels = client_1_inputs.to(device), client_1_labels.to(device)
        client_1_grad_dict = get_client_grad(client_1_inputs, client_1_labels, net_dict, client_1_net)

        client_average_grad_dict = dict()
        for key in client_0_grad_dict:
            client_average_grad_dict[key] = client_0_grad_dict[key] / CLIENT_NUM + client_1_grad_dict[key] / CLIENT_NUM

        params_modules_server = net.to(device).named_parameters()
        for params_module in params_modules_server:
            name, params = params_module
            params.grad = client_average_grad_dict[name]
        optimizer_server.step()

    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader0:
            images, labels = data
            images = torch.index_select(images, 1, torch.LongTensor([0]))
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('Epoch %d Acc 0: %.2f%%' % (epoch + 1, (100 * float(correct) / total)))
        
        correct = 0
        total = 0
        for data in testloader1:
            images, labels = data
            images = torch.index_select(images, 1, torch.LongTensor([0]))
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('Epoch %d Acc 1: %.2f%%' % (epoch + 1, (100 * float(correct) / total)))

print('Train Time: %.2f s/epoch' % ((time.time() - st) / EPOCH_NUM))

#torch.save(net.state_dict(), 'models/jointly_learning_demo_%d.pth' % (epoch + 1))
#print('successfully save the model to models/jointly_learning_demo_%d.pth' % (epoch + 1))
