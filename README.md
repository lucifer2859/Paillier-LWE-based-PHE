# Paillier-LWE-based-PHE
Privacy-Preserving Deep Learning via Additively Homomorphic Encryption

### CUDA版本：9.0与9.1皆可
### PyTorch: 1.1.0

## python-paillier-master
#### python setup.py test
#### python test.py

## LWE-based PHE
#### mkdir key // 创建用于存储密钥的文件夹
#### python cpu_test.py | python cuda_test.py

## 数据集分割
#### python split_data.py

## LeNet训练
#### mkdir models // 创建用于存储模型的文件夹
#### LeNet.py、LeNet_subset.py与jointly_learning_demo.py可以独立运行（注意数据集路径）
#### jointly_learning_with_encryption_demo.py需要注意import LeNet与cuda_test时的路径（可以将其与LeNet.py放入LWE-based PHE根目录中）

## 代码版本区别
### jointly_learning: 上传梯度，下载聚合后的梯度
#### v1: IID，平衡，数据集没有实际分割，在训练时模拟分割；
#### v2: IID，平衡，数据集有实际分割；
#### v3: IID，将用户进行了类封装，每个epoch迭代最小数据集的iteration数，逐用户测试；
#### v4: 在v3基础上添加了对Non-IID设置以及完整数据集测试的支持；
### federated_learning: 上传模型，下载聚合后的模型
#### v1: FedAvg;
#### v2: EASGD(Elastic Averaging SGD);
#### v3: FedProx;
