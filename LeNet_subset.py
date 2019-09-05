import torch
import torchvision as tv
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import platform
import copy
import time

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(), 
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda")
    torch.cuda.set_device(6)

    EPOCH_NUM = 100
    BATCH_SIZE = 64
    LR = 0.001

    transform = transforms.ToTensor()

    trainset = datasets.ImageFolder('/home/dchen/dataset/MNIST/IID/2/train/client0/', transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    testset = datasets.ImageFolder('/home/dchen/dataset/MNIST/IID/2/test/client0/', transform=transform)

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    net = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

    st = time.time()

    for epoch in range(EPOCH_NUM):
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = torch.index_select(inputs, 1, torch.LongTensor([0]))
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                images = torch.index_select(images, 1, torch.LongTensor([0]))
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('Epoch %d Acc: %.2f%%' % (epoch + 1, (100 * float(correct) / total)))
    
    print('Train Time: %.2f s/epoch' % ((time.time() - st) / EPOCH_NUM))

    #torch.save(net.state_dict(), 'models/lenet_%d.pth' % (epoch + 1))
    #print('successfully save the model to models/lenet_%d.pth' % (epoch + 1))
