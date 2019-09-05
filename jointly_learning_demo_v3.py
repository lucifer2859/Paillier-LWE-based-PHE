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

    def get_grad(self, client_inputs, client_labels, net_dict):
        self.net.load_state_dict(net_dict)
        client_outputs = self.net(client_inputs)
        client_loss = self.criterion(client_outputs, client_labels)
        client_optimizer = optim.SGD(self.net.parameters(), lr=LR, momentum=0.9)
        client_optimizer.zero_grad()
        client_loss.backward()

        client_grad_dict = dict()
        params_modules = list(self.net.named_parameters())
        for params_module in params_modules:
            name, params = params_module
            params_grad = copy.deepcopy(params.grad)
            client_grad_dict[name] = params_grad
        client_optimizer.zero_grad()
        return client_grad_dict

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

net = LeNet()
net.apply(weight_init)

optimizer_server = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

client_list = []
train_data_root = '/home/dchen/dataset/MNIST/IID/' + str(CLIENT_NUM) + '/train/'
test_data_root  = '/home/dchen/dataset/MNIST/IID/' + str(CLIENT_NUM) + '/test/'

for i in range(CLIENT_NUM):
    client_name = 'client' + str(i)
    client_list.append(Client(client_name, train_data_root + client_name + '/', test_data_root + client_name + '/'))

min_dataset_len = client_list[0].dataset_len
for i in range(1, CLIENT_NUM):
    if client_list[i].dataset_len < min_dataset_len:
        min_dataset_len = client_list[i].dataset_len

st = time.time()

for epoch in range(EPOCH_NUM):
    data_iter_list = []
    for i in range(CLIENT_NUM):
        data_iter_list.append(iter(client_list[i].trainloader))
    
    for index in range(min_dataset_len):
        net_dict = net.state_dict()

        client_grad_dict_list = []
        for i in range(CLIENT_NUM):
            client_inputs, client_labels = next(data_iter_list[i])
            client_inputs = torch.index_select(client_inputs, 1, torch.LongTensor([0]))
            client_inputs, client_labels = client_inputs.to(device), client_labels.to(device)
            client_grad_dict_list.append(client_list[i].get_grad(client_inputs, client_labels, net_dict))

        client_average_grad_dict = client_grad_dict_list[0]
        for i in range(1, CLIENT_NUM):    
            for key in client_grad_dict_list[0]:
                client_average_grad_dict[key] += client_grad_dict_list[i][key]
        for key in client_grad_dict_list[0]:
            client_average_grad_dict[key] /= CLIENT_NUM

        params_modules_server = net.to(device).named_parameters()
        for params_module in params_modules_server:
            name, params = params_module
            params.grad = client_average_grad_dict[name]
        optimizer_server.step()

    with torch.no_grad():
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
        

print('Train Time: %.2f s/epoch' % ((time.time() - st) / EPOCH_NUM))

#torch.save(net.state_dict(), 'models/jointly_learning_demo_%d.pth' % (epoch + 1))
#print('successfully save the model to models/jointly_learning_demo_%d.pth' % (epoch + 1))
