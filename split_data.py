import os
from skimage import io
import torchvision.datasets.mnist as mnist
import numpy

root = "/home/dchen/dataset/MNIST/raw/"
class_num = 10

train_set = (
  mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
  mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
)
  
test_set = (
  mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
  mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
)

print('---------------------------------------')
print("train image set:", train_set[0].size())
print("train label set:", train_set[1].size())
print("test iamge set:", test_set[0].size())
print("test label set:", test_set[1].size())
print('---------------------------------------')

def split_class(train=True):
  if(train):
    data_path = root + 'train/'
    if(not os.path.exists(data_path)):
      os.makedirs(data_path)
    
    os.chdir(data_path)
    os.system('rm -rf *')
    
    for i in range(class_num):
      class_path = data_path + str(i) + '/'
      if(not os.path.exists(class_path)):
        os.makedirs(class_path)
        
    for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
      img_path = data_path + str(train_set[1][i].item()) + '/' + str(i) + '.png'
      io.imsave(img_path, img.numpy())
  else:
    data_path = root + 'test/'
    if (not os.path.exists(data_path)):
      os.makedirs(data_path)
    
    os.chdir(data_path)
    os.system('rm -rf *')
    
    for i in range(class_num):
      class_path = data_path + str(i) + '/'
      if(not os.path.exists(class_path)):
        os.makedirs(class_path)
        
    for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
      img_path = data_path + str(test_set[1][i].item()) + '/' + str(i) + '.png'
      io.imsave(img_path, img.numpy())

def generate_IID_dataset(client_num, train=True):
  if(train):
    data_path = root + '../IID/train/'
    if(not os.path.exists(data_path)):
      os.makedirs(data_path)
    
    os.chdir(data_path)
    os.system('rm -rf *')
    
    for i in range(client_num):
      client_path = data_path + 'client' + str(i) + '/'
      if(not os.path.exists(client_path)):
        os.makedirs(client_path)
      
      for j in range(class_num):
        class_path = client_path + str(j) + '/'
        if(not os.path.exists(class_path)):
          os.makedirs(class_path)
      
    for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
      img_path = data_path + 'client' + str(i % client_num) + '/' + str(train_set[1][i].item()) + '/' + str(i) + '.png'
      io.imsave(img_path, img.numpy())
  else:
    data_path = root + '../IID/test/'
    if (not os.path.exists(data_path)):
      os.makedirs(data_path)
    
    os.chdir(data_path)
    os.system('rm -rf *')
    
    for i in range(client_num):
      client_path = data_path + 'client' + str(i) + '/'
      if(not os.path.exists(client_path)):
        os.makedirs(client_path)
      
      for j in range(class_num):
        class_path = client_path + str(j) + '/'
        if(not os.path.exists(class_path)):
          os.makedirs(class_path)
    
    for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
      img_path = data_path + 'client' + str(i % client_num) + '/' + str(test_set[1][i].item()) + '/' + str(i) + '.png'
      io.imsave(img_path, img.numpy())

def generate_Non_IID1_dataset(client_num, train=True):
  if(train):
    data_path = root + '../Non-IID1/' + str(client_num) + '/train/'
    if(not os.path.exists(data_path)):
      os.makedirs(data_path)
    
    os.chdir(data_path)
    os.system('rm -rf *')
    
    for i in range(client_num):
      client_path = data_path + 'client' + str(i) + '/'
      if(not os.path.exists(client_path)):
        os.makedirs(client_path)
        
      for j in range(class_num):
        class_path = client_path + str(j) + '/'
        if(not os.path.exists(class_path)):
          os.makedirs(class_path)
      
    for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
      img_path = data_path + 'client' + str(train_set[1][i].item() % client_num) + '/' + str(train_set[1][i].item()) + '/' + str(i) + '.png'
      io.imsave(img_path, img.numpy())
  else:
    data_path = root + '../Non-IID1/' + str(client_num) + '/test/'
    if (not os.path.exists(data_path)):
      os.makedirs(data_path)
    
    os.chdir(data_path)
    os.system('rm -rf *')
    
    for i in range(client_num):
      client_path = data_path + 'client' + str(i) + '/'
      if(not os.path.exists(client_path)):
        os.makedirs(client_path)
        
      for j in range(class_num):
        class_path = client_path + str(j) + '/'
        if(not os.path.exists(class_path)):
          os.makedirs(class_path)
    
    for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
      img_path = data_path + 'client' + str(test_set[1][i].item() % client_num) + '/' + str(test_set[1][i].item()) + '/' + str(i) + '.png'
      io.imsave(img_path, img.numpy())

'''
# Dataset Conversion
split_class(True)
print('split_class train')
split_class(False)
print('split_class test')

# 2 client, IID
generate_IID_dataset(2, True)
print('generate_IID_dataset train')
generate_IID_dataset(2, False)
print('generate_IID_dataset test')

# 10 client, 1 class per client, Non IID
generate_Non_IID1_dataset(10, True)
print('generate_Non_IID1_dataset train')
generate_Non_IID1_dataset(10, False)
print('generate_Non_IID1_dataset test')
'''

# 2 client, 5 class per client, Non IID
generate_Non_IID1_dataset(2, True)
print('generate_Non_IID1_dataset train')
generate_Non_IID1_dataset(2, False)
print('generate_Non_IID1_dataset test')
