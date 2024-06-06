import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms


def load_mnist(batch_size):
    # 初始化MNIST数据集加载器的函数

    # 超参数定义部分
    # 输入图像的尺寸是28x28像素，因此输入尺寸为28
    input_size = 28
    # MNIST数据集包含10个不同的类别（数字0-9）
    num_classes = 10
    # 训练模型时将完整遍历数据集的次数设为3次
    num_epochs = 3
    # 指定每个批次处理的数据量为64（这里直接使用函数参数batch_size更灵活）
    # 注意：下面代码中实际使用的batch_size会以函数参数为准

    # 加载MNIST训练集
    # 参数说明：
    # root: 数据存储的根目录
    # train: 设为True表示加载训练数据集
    # transform: 将图像数据转化为PyTorch的Tensor格式
    # download: 如果数据不存在则自动下载
    train_dataset = datasets.MNIST(
        root="../data",
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )

    # 加载MNIST测试集
    # 参数与训练集相似，只是train设为False来加载测试数据
    test_dataset = datasets.MNIST(
        root="../data", train=False, transform=transforms.ToTensor()
    )

    # 使用DataLoader创建数据加载器，它会在训练时按批次提供数据
    # shuffle=True表示在每次训练之前都会随机打乱数据顺序，有助于训练过程的稳定性和泛化能力
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    # 同样为测试集创建数据加载器
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True
    )

    # 函数最后返回训练和测试数据加载器，方便后续模型训练和评估使用
    return train_loader, test_loader


# 定义一个卷积神经网络（CNN）类，继承自nn.Module
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # 调用父类的构造函数初始化
        # 定义第一个卷积层块
        self.conv1 = nn.Sequential(  # Sequential表示顺序容器
            nn.Conv2d(
                in_channels=1,  # 输入通道数，灰度图是1通道
                out_channels=16,  # 输出通道数（特征图数）为16
                kernel_size=5,  # 卷积核大小为5x5
                stride=1,  # 卷积步长为1
                padding=2,  # 填充为2，使输出与输入大小相同
            ),  # 卷积后输出的特征图大小为 (16, 28, 28)
            nn.ReLU(),  # ReLU激活函数，使非线性
            nn.MaxPool2d(
                kernel_size=2
            ),  # 最大池化，池化核大小为2x2，输出大小变为 (16, 14, 14)
        )
        # 定义第二个卷积层块
        self.conv2 = nn.Sequential(  # 输入特征图大小为 (16, 14, 14)
            nn.Conv2d(
                16, 32, 5, 1, 2
            ),  # 输入通道16，输出通道32，卷积核5x5，步长1，填充2
            nn.ReLU(),  # ReLU激活函数
            nn.Conv2d(32, 32, 5, 1, 2),  # 再次卷积，保持输入输出通道数相同
            nn.ReLU(),  # ReLU激活函数
            nn.MaxPool2d(2),  # 最大池化，池化核2x2，输出大小变为 (32, 7, 7)
        )
        # 定义第三个卷积层模块：继续深化特征学习
        self.conv3 = nn.Sequential(
            # 上一层输出是32个特征图，每个大小为14x14像素
            nn.Conv2d(
                in_channels=32,  # 接收来自上一层的32个特征图
                out_channels=64,  # 本层输出将增强到64个特征图，有助于捕捉更复杂的特征
                kernel_size=5,  # 使用5x5的卷积核探索局部特征
                stride=1,  # 步长为1，意味着卷积核在特征图上逐像素移动
                padding=2,  # 填充2个像素，保持输出特征图尺寸与输入相同，这里是(14, 14)
            ),
            nn.ReLU(),  # 应用ReLU激活函数，增加网络的非线性表达能力，激活输出特征
            # 注意：此处未再进行池化操作，保持空间尺寸为(64, 14, 14)，更多关注特征的丰富而非下采样
        )

        # 定义全连接层
        self.out = nn.Linear(
            64 * 7 * 7, 10
        )  # 全连接层，将64*7*7个节点连接到10个输出节点（类别数为10）

    # 定义前向传播过程
    def forward(self, x):
        x = self.conv1(x)  # 输入数据经过第一个卷积层块
        x = self.conv2(x)  # 然后经过第二个卷积层块
        x = self.conv3(x)  # 再经过第三个卷积层块
        x = x.view(
            x.size(0), -1
        )  # 将多维的特征图展平成一维向量，形状为 (batch_size, 64*7*7)
        output = self.out(x)  # 输入到全连接层，得到最终的输出
        return output  # 返回输出结果


def accuracy(predictions, labels):
    """
    计算模型预测的准确率，即预测正确的样本数占总样本数的比例。
    - predictions: 这是一个来自PyTorch的张量（tensor），包含了模型对每个样本的预测得分。每个样本对应一列，每列中的数值表示该样本属于各个类别的可能性大小。
    - labels: 同样是PyTorch张量，但这里存储的是每个样本的真实类别标签。它的形状应与predictions相对应，确保一一匹配。
    返回值说明：
    - rights: 一个整数，代表预测正确的样本数量。通过比较预测类别和实际类别得到。
    - total: 一个整数，表示总共有多少个样本参与了这次准确率的计算。
    """
    # 使用torch.max找到predictions中每行的最大值及对应的索引，索引即为预测的类别
    pred = torch.max(predictions.data, 1)[1]

    # 将预测类别pred与实际类别labels进行元素级比较，eq函数会返回一个布尔型张量，True表示预测正确，False表示预测错误。
    # 使用sum函数累计预测正确的数量（True被视为1，False被视为0）
    rights = pred.eq(labels.data.view_as(pred)).sum()

    # 返回预测正确的数量和总样本数
    return rights, len(labels)


def train_model(
        net, train_loader, test_loader, num_epochs, batch_size, learning_rate=0.001
):
    """
    参数：
    net (nn.Module): 待训练的神经网络
    train_loader (DataLoader): 训练数据加载器
    test_loader (DataLoader): 测试数据加载器
    num_epochs (int): 训练轮数
    batch_size (int): 每批数据大小
    learning_rate (float): 学习率，默认值为0.001
    """

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器，这里使用Adam优化算法
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # 开始训练循环
    for epoch in range(num_epochs):
        # 保存当前epoch的结果
        train_rights = []

        # 针对训练数据加载器中的每一个批进行循环
        for batch_idx, (data, target) in enumerate(train_loader):
            net.train()  # 将模型设置为训练模式
            output = net(data)  # 前向传播，获取模型输出
            loss = criterion(output, target)  # 计算损失
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新参数
            right = accuracy(output, target)  # 计算准确率
            train_rights.append(right)  # 保存每批的准确率

            # 每经过100个批次，进行一次验证
            if batch_idx % 100 == 0:
                net.eval()  # 将模型设置为评估模式
                val_rights = []  # 保存验证结果

                # 针对验证数据加载器中的每一个批进行循环
                for data, target in test_loader:
                    output = net(data)  # 前向传播，获取模型输出
                    right = accuracy(output, target)  # 计算准确率
                    val_rights.append(right)  # 保存每批的准确率

                # 计算训练集和验证集的准确率
                # 计算训练集和验证集的总正确预测数以及总样本数
                # 对于每个批次，train_rights和val_rights列表分别存储了该批次的正确预测数和总样本数
                # tup[0]表示正确预测数，tup[1]表示总样本数
                train_total_correct = sum(
                    [tup[0] for tup in train_rights]
                )  # 训练集总正确预测数
                train_total_samples = sum(
                    [tup[1] for tup in train_rights]
                )  # 训练集总样本数
                val_total_correct = sum(
                    [tup[0] for tup in val_rights]
                )  # 验证集总正确预测数
                val_total_samples = sum(
                    [tup[1] for tup in val_rights]
                )  # 验证集总样本数

                # 使用元组存储这些值，方便后续计算准确率
                train_r = (train_total_correct, train_total_samples)
                val_r = (val_total_correct, val_total_samples)

                # 打印当前训练进度和性能指标，帮助初学者直观了解训练情况
                # epoch: 当前正在进行的训练轮次
                # [batch_idx * batch_size/{len(train_loader.dataset)}: 表示已完成的样本数/总样本数
                # ({:.0f}%): 完成百分比，计算已完成的批次占总批次的比例
                # 损失: {:.6f}: 当前批次的平均损失值，衡量模型预测错误的程度
                # 训练集准确率: {:.2f}%: 根据train_r计算得出，表示训练集上模型预测正确的百分比
                # 测试集正确率: {:.2f}%: 根据val_r计算得出，表示验证集上模型预测正确的百分比

                print(
                    "当前epoch: {} [{}/{} ({:.0f}%)]\t损失: {:.6f}\t训练集准确率: {:.2f}%\t测试集正确率: {:.2f}%".format(
                        epoch,  # 当前训练轮数
                        batch_idx * batch_size,  # 当前批次结束时已处理的样本数
                        len(train_loader.dataset),  # 总样本数
                        100.0 * batch_idx / len(train_loader),  # 完成的百分比
                        loss.data,  # 当前批次的损失值
                        100.0 * train_r[0] / train_r[1],  # 训练集准确率
                        100.0 * val_r[0] / val_r[1],  # 验证集准确率
                    )
                )


if __name__ == "__main__":
    # 加载MNIST数据集，每个批次64个样本
    train_loader, test_loader = load_mnist(batch_size=64)

    # 实例化一个卷积神经网络模型
    net = CNN()

    # 调用函数训练模型，传入网络模型、训练数据加载器、验证数据加载器、训练轮数和批次大小
    train_model(net, train_loader, test_loader, num_epochs=2, batch_size=64)
