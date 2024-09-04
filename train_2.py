import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import ResNet18
from torch.utils.tensorboard import SummaryWriter
import os
from utils import ConfusionMatrix
from tqdm import tqdm

import time
import psutil
import torch
import torch.optim.lr_scheduler as lr_scheduler
# 定义是否使用GPU
device = torch.device("cuda")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
args = parser.parse_args()

# 超参数设置
EPOCH = 330
   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 128      #批处理尺寸(batch_size)
LR = 0.1      #学习率



# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train) #训练数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型定义-ResNet
net = ResNet18().to(device)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题，封装了softmax
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [136,186,280,330], gamma=0.1) #按设定的间隔调整学习率

# 训练
if __name__ == "__main__":
    best_acc = 85  #2 初始化best test accuracy
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    # 初始化 TensorBoard writer
    writer = SummaryWriter(log_dir='./logs')


    # 检查是否需要恢复训练
    if args.net:
        if os.path.isfile(args.net):
            print(f"Loading checkpoint '{args.net}'")
            checkpoint = torch.load(args.net)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            pre_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            print(f"Checkpoint loaded. Resuming from epoch {pre_epoch}")
             # 手动调整学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR
            print(f"Learning rate manually set to {LR}")
        else:
            print(f"No checkpoint found at '{args.net}'")

    # 记录训练开始时间
    start_time = time.time()
    # 获取初始的内存和CPU使用情况
    initial_memory = psutil.virtual_memory().used
    initial_cpu = psutil.cpu_percent(interval=1)
    
    with open("acc.txt", "a") as f:
        with open("log.txt", "a")as f2:
            for epoch in range(pre_epoch, EPOCH):
                scheduler.step()
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0

                # 使用 tqdm 包装 trainloader
                with tqdm(enumerate(trainloader, 0), total=len(trainloader), desc="Training",ncols=180) as pbar:
                    for i, data in pbar:
                        # 准备数据
                        length = len(trainloader)
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        optimizer.zero_grad()

                        # forward + backward
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        # 每训练1个batch打印一次loss和准确率
                        sum_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += predicted.eq(labels.data).cpu().sum()
                        # 获取资源使用情况
                        cpu_usage = psutil.cpu_percent(interval=None)  # 获取当前CPU使用率
                        memory_info = psutil.virtual_memory()  # 获取内存使用情况
                        gpu_memory = torch.cuda.memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0
                        gpu_usage = torch.cuda.memory_reserved(device) / (1024 ** 2) if torch.cuda.is_available() else 0
                    
                        # 更新 tqdm 描述
                        #pbar.set_postfix(loss=sum_loss / (i + 1), acc=100. * correct / total)
                        
                        pbar.set_postfix({
                            'loss': f'{sum_loss / (i + 1):.3f}', 
                            'acc': f'{100. * correct / total:.3f}%', 
                            'CPU': f'{cpu_usage}%', 
                            'Mem': f'{memory_info.percent}%', 
                            'GPU Mem': f'{gpu_memory:.2f}MB', 
                            'GPU Res': f'{gpu_usage:.2f}MB'
                        })
                        # 每个 batch 记录损失和准确率到 TensorBoard
                        writer.add_scalar('Train/Loss', loss.item(), epoch * len(trainloader) + i)
                        writer.add_scalar('Train/Accuracy', 100. * correct / total, epoch * len(trainloader) + i)
                        
                        #print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                        #      % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                        f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                            % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                        f2.write('\n')
                        f2.flush()

                for param_group in optimizer.param_groups:
                    print(f"当前学习率: {param_group['lr']}")
                
                # 每训练完一个epoch测试一下准确率
                #创建混淆矩阵
                labels = [label for label in classes]
                confusion = ConfusionMatrix(num_classes=10, labels=labels)

                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()  #指明是在测试
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        # 更新混淆矩阵的值
                        confusion.update(predicted.to("cpu").numpy(), labels.to("cpu").numpy()) 
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total

                    writer.add_scalar('Test/Accuracy', acc, epoch)

                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    #序列化保存模型
                    #torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_acc': best_acc
                    }, '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
                
                #confusion.plot()         # 绘制混淆矩阵
                confusion.summary()      # 计算指标

            print("Training Finished, TotalEPOCH=%d" % EPOCH)

            # 关闭 TensorBoard writer
            writer.close()
