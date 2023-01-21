import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import MyLeNet

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 指定设备

    # 1. 准备数据集(自定义数据集见01_customized_data.py)
    # 1.1 加载数据集
    train_data = torchvision.datasets.CIFAR10("./CIFAR10_dataset", train=True, download=True,
                                              transform=torchvision.transforms.ToTensor())
    test_data = torchvision.datasets.CIFAR10("./CIFAR10_dataset", train=False, download=True,
                                             transform=torchvision.transforms.ToTensor())

    # 1.2 划分验证集
    val_percent = 0.1
    val_size = int(val_percent * len(train_data))
    train_size = len(train_data) - val_size
    train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

    print("训练集大小为:{0}, 测试集大小为:{1}, 验证集大小为{2}".format(len(train_data), len(test_data), len(val_data)))

    # 2. 用DataLoader加载数据集
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)

    # 3. 搭建网络(类定义见my_model.py) [GPU]
    model = MyLeNet()
    # model.load_state_dict(torch.load("my_neural_network.pth"))
    model = model.to(device)  # GPU Train

    # 4. 损失函数 [GPU]
    loss_fn = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_fn = loss_fn.to(device)  # GPU Train

    # 5. 优化器
    learning_rate = 1e-2  # 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 6. 设置训练所需的一些参数
    total_train_step = 0
    epochs = 50

    # 6+. tensorboard
    writer = SummaryWriter("./logs_train")

    # 7. 开始训练 [GPU用在数据上]
    for epoch in range(epochs):
        print("[-------第{}轮训练开始-------]".format(epoch + 1))

        # 7.1 开始训练
        model.train()  # 训练模式
        for data in train_dataloader:
            # 7.1.1 正向计算
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            # 7.1.2 反向传播
            optimizer.zero_grad()  # 把累加的梯度清空
            loss.backward()  # 反向传播
            optimizer.step()  # 优化参数

            # 7.1.3 数据记录
            total_train_step += 1
            if total_train_step % 100 == 0:
                print("训练次数:{0}, Loss:{1}".format(total_train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), total_train_step)

        # 7.2 开始验证 防止过拟合
        model.eval()  # 开启评估模式
        total_val_loss = 0  # 验证集上的Σ损失值
        total_correct = 0  # 验证集上预测正确的次数
        with torch.no_grad():
            for data in val_dataloader:
                # 7.2.1 正向计算
                imgs, labels = data
                if torch.cuda.is_available():
                    imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)

                # 7.2.2 数据记录
                total_val_loss += loss
                total_correct += (outputs.argmax(1) == labels).sum()  # 标签与猜测的结果相同的数量

        total_accuracy = total_correct / val_size  # 验证集上预测的准确率
        print("验证集总体 Loss: {}".format(total_val_loss))
        writer.add_scalar("val_loss", total_val_loss, epoch)
        print("验证集总体 Accuracy: {}".format(total_accuracy))
        writer.add_scalar("val_accuracy", total_accuracy, epoch)

        # 7.4 每一轮训练后保存结果
        torch.save(model.state_dict(), "my_neural_network.pth")
        print("模型已保存...")

    # 8. 最终测试
    print("[-------最终测试-------]")
    total_test_loss = 0
    total_test_correct = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, labels = data
            if torch.cuda.is_available():
                imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            total_test_loss += loss
            total_test_correct += (outputs.argmax(1) == labels).sum()

    total_accuracy = total_test_correct / len(test_data)
    print("测试集总体 Loss: {}".format(total_test_loss))
    print("测试集总体 Accuracy: {}".format(total_accuracy))
    writer.close()
