# EASGD(Elastic Averaging SGD)

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

ROUND_NUM = 2000
LOCAL_EPOCH_NUM = 1
BATCH_SIZE = 10
LR = 0.01
CLIENT_NUM = 10

MU = 1

device = torch.device("cuda")
torch.cuda.set_device(6)

class Client:
    def __init__(self, name, train_data_dir, test_data_dir):
        self.name = name

        transform = transforms.ToTensor()

        trainset = datasets.ImageFolder(train_data_dir, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        testset = datasets.ImageFolder(test_data_dir, transform=transform)
        self.testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        dataset_list = list(self.trainloader)
        self.dataset_len = len(dataset_list)

        self.net = LeNet().to(device)

        self.criterion = nn.CrossEntropyLoss()

    def update(self, net_dict, center_params_dict):
        self.net.load_state_dict(net_dict)

        for i in range(LOCAL_EPOCH_NUM):
            data_iter = iter(self.trainloader)
            for b in range(self.dataset_len):
                inputs, labels = next(data_iter)
                inputs = torch.index_select(inputs, 1, torch.LongTensor([0]))
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                optimizer = optim.SGD(self.net.parameters(), lr=LR, momentum=0.9)
                optimizer.zero_grad()
                loss.backward()

                params_modules = list(self.net.named_parameters())
                for params_module in params_modules:
                    name, params = params_module
                    params.grad += MU * (params.data - center_params_dict[name])

                optimizer.step()

        return self.net.state_dict()


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

net = LeNet().to(device)
net.apply(weight_init)

optimizer_server = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

client_list = []
# train_data_root = '/home/dchen/dataset/MNIST/IID/' + str(CLIENT_NUM) + '/train/'
# test_data_root  = '/home/dchen/dataset/MNIST/IID/' + str(CLIENT_NUM) + '/test/'
train_data_root = '/home/dchen/dataset/MNIST/Non-IID1/' + str(CLIENT_NUM) + '/train/'
test_data_root  = '/home/dchen/dataset/MNIST/Non-IID1/' + str(CLIENT_NUM) + '/test/'

for i in range(CLIENT_NUM):
    client_name = 'client' + str(i)
    client_list.append(Client(client_name, train_data_root + client_name + '/', test_data_root + client_name + '/'))

center_params_dict = dict()
center_params_modules = list(self.net.named_parameters())
for params_module in center_params_modules:
    name, params = params_module
    center_params_dict[name] = copy.deepcopy(params.data)

st = time.time()

for t in range(ROUND_NUM):
    client_net_dict_list = []
    net_dict = net.state_dict()

    for i in range(CLIENT_NUM):
        client_net_dict_list.append(client_list[i].update(net_dict, center_params_dict))

    client_average_net_dict = client_net_dict_list[0]
    for key in client_average_net_dict:
        for i in range(1, CLIENT_NUM):
            client_average_net_dict[key] += client_net_dict_list[i][key]
    for key in client_net_dict_list[0]:
        client_average_net_dict[key] /= CLIENT_NUM

    net.load_state_dict(client_average_net_dict)

    for key in center_params_dict:
        tmp_params = center_params_dict[key]
        for i in range(CLIENT_NUM):
            center_params_dict[key] += LR * MU * (client_net_dict_list[i][key] - tmp_params)

    with torch.no_grad():
        '''
        # test per client
        for i in range(CLIENT_NUM):
            correct = 0
            total = 0
            for data in client_list[i].testloader:
                images, labels = data
                images = torch.index_select(images, 1, torch.LongTensor([0]))
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('Epoch %d Acc (%s): %.2f%%' % (epoch + 1, client_list[i].name, (100 * float(correct) / total)))
        '''
        
        # test all
        correct = 0
        total = 0
        for i in range(CLIENT_NUM):    
            for data in client_list[i].testloader:
                images, labels = data
                images = torch.index_select(images, 1, torch.LongTensor([0]))
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
        print('Round %d Acc: %.2f%%' % (t + 1, (100 * float(correct) / total)))
        

print('Train Time: %.2f s/round' % ((time.time() - st) / ROUND_NUM))

#torch.save(net.state_dict(), 'models/federated_learning_demo_%d.pth' % (round + 1))
#print('successfully save the model to models/federated_learning_demo_%d.pth' % (round + 1))