# LeNet 网络
################################
# 层名            大小      参数数目
# input           1*28*28   
# conv1           6*28*28   150+6=156  
# maxpool         6*14*14   
# conv2           16*10*10  2400+16=2416
# maxpool         16*5*5
# linear          120       48000+120=48120
# linear          84        10080+84=10164
# linear(output)  10        840+10=850
# 参数大小: 61706*4B=241KB
###################################
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import platform
import copy
import time

# 定义网络结构
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(     #input_size=(1*28*28)
            #(in_channels, out_channels, kernel_size, stride=1, padding=0)
            nn.Conv2d(1, 6, 5, 1, 2), #padding=2保证输入输出尺寸相同
            nn.ReLU(),      #input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2)   #output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),      #input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
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

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# 训练
if __name__ == "__main__":
    # 定义是否使用GPU
    device = torch.device("cuda")
    torch.cuda.set_device(6)

    # 超参数设置
    EPOCH_NUM = 100  # 遍历数据集次数
    BATCH_SIZE = 64  # 批处理尺寸(batch_size)
    LR = 0.001  # 学习率

    # 定义数据预处理方式
    transform = transforms.ToTensor()

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

    # 定义损失函数loss function 和优化方式（采用SGD）
    net = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

    st = time.time()

    for epoch in range(EPOCH_NUM):
        #sum_loss = 0.0
        # 数据读取
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            '''
            # 每训练100个batch打印一次平均loss
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %d] loss: %.3f'
                      % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
            '''
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
    
    print('Train Time: %.2f s/epoch' % ((time.time() - st) / EPOCH_NUM))

    parameter_num = 0
    print('模型参数数目：')
    model_parameters = net.state_dict()
    for key, value in model_parameters.items():
        parameter_num += torch.numel(value)
        print("%s: %d" % (key, torch.numel(value)))
    print('------------------------------------------------------')
    print("Total: %d, Size: %.2fKB" % (parameter_num, float(parameter_num) * 4 / 1024))
    print('------------------------------------------------------')

    #torch.save(net.state_dict(), 'models/lenet_%d.pth' % (epoch + 1))
    #print('successfully save the model to models/lenet_%d.pth' % (epoch + 1))
