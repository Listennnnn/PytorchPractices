import torch  # PyTorch 主库
from torch import (  # 从 PyTorch 导入具体模块
    nn,  # 常用的神经网络模块
)
from pathlib import Path  # 用于路径操作的模块
import pickle  # 用于数据序列化和反序列化的模块
import gzip  # 用于文件压缩的模块
import torch.nn.functional as F  # PyTorch 中的神经网络功能函数
from torch.utils.data import TensorDataset, DataLoader  # PyTorch 的数据处理工具

import numpy as np  # testremote3

from torch import optim  # PyTorch 的优化器模块


def get_model():
    """
    创建并返回一个Mnist_NN模型实例及其优化器。

    参数:
    无

    返回:
    model: Mnist_NN模型实例，用于图像分类。
    optim.Adam: 优化器实例，使用Adam算法优化模型参数，学习率为0.001。
    """
    model = Mnist_NN()  # 创建Mnist_NN模型实例
    return model, optim.Adam(model.parameters(), lr=0.001)  # 返回模型和配置好的优化器


def loss_batch(model, loss_func, xb, yb, opt=None):
    """
    计算并返回给定模型在一个批次数据上的损失值。

    参数:
    - model: 训练的模型，用于预测。
    - loss_func: 用于计算损失的函数。
    - xb: 批次的输入数据。
    - yb: 批次的真实标签。
    - opt: 优化器，如果提供，则进行反向传播和参数更新。

    返回:
    - 一个元组，包含批次的损失值和输入数据的大小。
    """

    # 计算损失
    loss = loss_func(model(xb), yb)

    # 如果提供了优化器，则进行反向传播、参数更新并重置梯度
    if opt is not None:
        loss.backward()
        opt.step()  # 更新模型参数
        opt.zero_grad()  # 重置梯度

    # 返回损失值和输入数据大小
    return loss.item(), len(xb)


def fit(steps, model, loss_func, opt, train_dl, valid_dl):
    """
    训练模型。

    参数:
    - steps: 训练步数。
    - model: 待训练的模型。
    - loss_func: 损失函数。
    - opt: 优化器。
    - train_dl: 训练数据加载器。
    - valid_dl: 验证数据加载器。

    不返回任何值，训练过程中打印当前步数和验证集损失。
    """
    for step in range(steps):
        # 进入训练模式并迭代训练数据集的每个批次
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        # 切换到评估模式并计算验证集上的损失
        model.eval()
        with torch.no_grad():
            # 计算验证集的平均损失
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        # 打印当前步数和验证集损失
        print("当前step:" + str(step), "验证集损失：" + str(val_loss))


def create_data_loaders(x_train, y_train, x_valid, y_valid, bs):
    """
    创建训练和验证数据加载器。

    参数:
    x_train (Tensor): 训练数据的特征部分
    y_train (Tensor): 训练数据的标签部分
    x_valid (Tensor): 验证数据的特征部分
    y_valid (Tensor): 验证数据的标签部分
    bs (int): 批量大小

    返回:
    train_dl (DataLoader): 训练数据的数据加载器
    valid_dl (DataLoader): 验证数据的数据加载器
    """
    net = Mnist_NN()  # 初始化Mnist神经网络模型

    train_ds = TensorDataset(x_train, y_train)  # 创建训练数据集
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)  # 创建训练数据加载器

    valid_ds = TensorDataset(x_valid, y_valid)  # 创建验证数据集
    valid_dl = DataLoader(
        valid_ds, batch_size=bs * 2
    )  # 创建验证数据加载器，批量大小为训练的两倍

    return train_dl, valid_dl


class Mnist_NN(nn.Module):
    """
    Mnist_NN 是一个用于手写数字识别的神经网络模型。该模型继承自nn.Module, 包含两个隐藏层和一个输出层。
    必须继承nn.Module且在其构造函数中需调用nn.Module的构造函数
    无需写反向传播函数，nn.Module能够利用autograd自动实现反向传播
    Module中的可学习参数可以通过named_parameters()或者parameters()返回迭代器
    """

    def __init__(self):
        """
        初始化Mnist_NN模型的结构。
        """
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)  # 第一个隐藏层，输入维度784，输出维度128
        self.hidden2 = nn.Linear(128, 256)  # 第二个隐藏层，输入维度128，输出维度256
        self.out = nn.Linear(256, 10)  # 输出层，输入维度256，输出维度10（对应10个类别）

    def forward(self, x):
        """
        定义Mnist_NN模型的前向传播路径。

        参数:
        - x : 输入数据张量

        返回值:
        - x : 经过网络处理后的输出数据张量
        """
        x = F.relu(self.hidden1(x))  # 通过第一个隐藏层和激活函数ReLU
        x = F.relu(self.hidden2(x))  # 通过第二个隐藏层和激活函数ReLU
        x = self.out(x)  # 通过输出层
        return x


def testGPU():
    """
    查询打印PyTorch及其CUDA相关配置的信息。
    """
    # 打印PyTorch版本
    print(torch.__version__)
    # 检查CUDA是否可用
    print(torch.cuda.is_available())
    # 打印可用的CUDA设备数量
    print(torch.cuda.device_count())
    # 打印第一个CUDA设备的名称
    print(torch.cuda.get_device_name(0))
    # 打印第一个CUDA设备的属性
    print(torch.cuda.get_device_properties(0))


# 读取数据
def load_data():
    """
    加载MNIST数据集。
    该函数从指定的本地路径"data/mnist"加载MNIST数据集。然后，使用pickle库解压缩并加载数据。

    返回值:
        x_train: 训练集图像数据
        y_train: 训练集标签
        x_valid: 验证集图像数据
        y_valid: 验证集标签

    注意: 数据集被分割为训练集和验证集，其中训练集有60000个样本，验证集有10000个样本。784是mnist数据集每个样本的像素点个数。
    """

    # 定义数据存储路径
    DATA_PATH = Path("../data")
    # 创建数据子目录，如果不存在的话
    PATH = DATA_PATH / "mnist"
    PATH.mkdir(parents=True, exist_ok=True)
    # 定义数据文件名
    FILENAME = "mnist.pkl.gz"

    # 使用gzip和pickle加载数据
    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        # 解压并加载pickle文件中的训练、验证数据和测试数据（我们仅使用训练和验证数据）
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

    return x_train, y_train, x_valid, y_valid


if __name__ == "__main__":
    # 取消下列语句注释，即可测试GPU是否可用
    # testGPU()

    # 加载数据
    x_train, y_train, x_valid, y_valid = load_data()

    # 数据需转换成tensor才能参与后续建模训练
    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )

    # 设置损失函数为交叉熵损失函数
    loss_func = F.cross_entropy

    # 定义批量大小
    bs = 64

    # - 一般在训练模型时加上model.train()，这样会正常使用Batch Normalization和Dropout
    # - 测试的时候一般选择model.eval()，这样就不会使用Batch Normalization和Dropout

    # 创建训练和验证数据加载器

    train_dl, valid_dl = create_data_loaders(x_train, y_train, x_valid, y_valid, bs)

    # 获取模型及其优化器。
    model, opt = get_model()

    """
    训练模型
    参数:
    - epochs: 训练的轮数。
    - model: 待训练的模型。
    - loss_func: 损失函数，用于度量模型的训练效果。
    - opt: 优化器，用于更新模型的参数。
    - train_dl: 训练数据加载器。
    - valid_dl: 验证数据加载器。
    """
    fit(20, model, loss_func, opt, train_dl, valid_dl)

    """
        计算并打印模型在验证集上的准确率
        model: 训练好的模型，用于对验证集数据进行预测
        valid_dl: 验证集的数据加载器，提供批量的验证样本
    """

    correct = 0  # 记录正确预测的数量
    total = 0  # 记录总预测数量

    # 遍历验证集中的每个样本
    for xb, yb in valid_dl:
        # 使用模型预测样本的类别
        outputs = model(xb)
        _, predicted = torch.max(outputs.data, 1)  # 获取预测的类别

        # 更新总预测数量和正确预测数量
        total += yb.size(0)
        correct += (predicted == yb).sum().item()

    # 计算并打印准确率
    print(
        "Accuracy of the network on the 10000 test images:%d%% "
        % (100 * correct / total)
    )
