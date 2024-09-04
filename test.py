import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import ResNet18
from utils import ConfusionMatrix
from tqdm import tqdm
import time
import psutil
import torch
from torchsummary import summary
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to load the trained model)")  # 加载训练好的模型路径
args = parser.parse_args()

# 准备数据集并预处理
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# 加载模型
net = ResNet18().to(device)

#scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
checkpoint = torch.load(args.net)
net.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#for param_group in optimizer.param_groups:
 #   param_group['lr'] = LR
net.eval()

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 测试模型
correct = 0
total = 0

# 创建混淆矩阵
confusion = ConfusionMatrix(num_classes=10, labels=classes)

print("Start Testing, Resnet-18!")
# 记录推理开始时间
start_inference_time = time.time()
# 获取初始的内存
initial_memory = psutil.virtual_memory().used

with torch.no_grad():
    start_time = time.time()
    for i, data in enumerate(tqdm(testloader, desc="Testing")):
        batch_start_time = time.time()
        
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        loss = criterion(outputs, labels)
        
        # 取得分最高的那个类 (outputs.data的索引号)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新混淆矩阵的值
        confusion.update(predicted.to("cpu").numpy(), labels.to("cpu").numpy()) 
        
        # 获取资源使用情况
        cpu_usage = psutil.cpu_percent(interval=None)  # 获取当前CPU使用率
        memory_info = psutil.virtual_memory()  # 获取内存使用情况
        gpu_memory = torch.cuda.memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0
        gpu_usage = torch.cuda.memory_reserved(device) / (1024 ** 2) if torch.cuda.is_available() else 0
        
        # 计算每个batch的推理时间
        batch_time = time.time() - batch_start_time
        
        # 仅在进度条外部打印硬件使用情况
        tqdm.write(f'Batch [{i+1}/{len(testloader)}]: Time: {batch_time:.2f}s | CPU Usage: {cpu_usage:.2f}% | '
                   f'Memory Usage: {memory_info.percent:.2f}% | GPU Memory Allocated: {gpu_memory:.2f}MB | '
                   f'GPU Memory Reserved: {gpu_usage:.2f}MB')

# 记录推理结束时间
end_inference_time = time.time()
# 获取训练结束时的内存
final_memory = psutil.virtual_memory().used


# 输出测试准确率
print('测试分类准确率为：%.3f%%' % (100. * correct / total))

# 显示混淆矩阵和其他指标
confusion.summary()

# 计算推理时间
inference_time = end_inference_time - start_inference_time
print(f"Inference Time: {inference_time} seconds")

# 计算资源使用情况
memory_used = (final_memory - initial_memory) / (1024 ** 3)  # 转换为 GB


print(f"Memory Used: {memory_used} GB")
# 打印模型的参数量
summary(net, (3, 32, 32))  # 这里 (3, 32, 32) 是 CIFAR-10 输入的维度


