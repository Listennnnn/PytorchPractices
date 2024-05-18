import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
import matplotlib.pyplot as plt


def load_data(filename):
    # 从相对路径加载CSV文件到DataFrame
    features = pd.read_csv(filename)
    # 打印数据的前五行，帮助我们理解数据格式
    print("数据的前几行：", features.head())
    # 打印数据的维度（行数和列数）
    print("数据维度：", features.shape)

    # 使用get_dummies将分类特征转换为数值表示
    features = pd.get_dummies(features)
    # 打印转换后的数据前五行
    print("转换后数据的前5行：", features.head(5))

    # 提取标签（目标变量），这里假设"actual"列是我们要预测的变量
    labels = np.array(features["actual"])
    print("打印标签的前5个值：", labels[:5])
    # 从特征数据中移除"actual"列，因为我们不再需要它
    features = features.drop("actual", axis=1)

    # 获取特征列名，用于后续参考
    feature_list = list(features.columns)
    print("特征列名：", feature_list)

    # 将DataFrame转换为numpy数组，因为许多机器学习模型需要这种格式
    features = np.array(features)
    print("转换后特征的形状：", features.shape)

    # 使用StandardScaler对输入特征进行标准化处理，使其均值为0，标准差为1
    input_features = preprocessing.StandardScaler().fit_transform(features)
    # 打印标准化处理后的第一个样本，以便观察变化
    print("标准化后的第一个样本：", input_features[0])

    return input_features, labels, feature_list


# 构建网络模型
def build_model(input_features, labels):
    """
    输入参数:
    input_features: 一个二维数组，包含训练数据的特征。
    labels: 一个一维数组，表示每个样本对应的标签。
    """

    # 获取输入特征的维度
    input_size = input_features.shape[1]

    # 输出只有一个值，所以输出层的维度是1
    output_size = 1

    # 设置隐藏层的节点数
    hidden_size = 128

    # 设置每批处理的样本数量
    batch_size = 16

    # 创建一个简单的神经网络，包含两个全连接层
    my_nn = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),  # 输入到隐藏层
        torch.nn.Sigmoid(),  # 隐藏层激活函数
        torch.nn.Linear(hidden_size, output_size),  # 隐藏层到输出层
    )

    # 定义损失函数，这里使用均方误差（MSE）
    cost = torch.nn.MSELoss(reduction="mean")

    # 使用Adam优化器进行参数更新
    optimizer = torch.optim.Adam(my_nn.parameters(), lr=0.001)

    # 训练神经网络
    losses = []  # 存储损失历史
    for epoch in range(1000):  # 进行1000次迭代
        batch_losses = []  # 存储每个批次的损失

        # 使用MINI-Batch方法进行训练
        for batch_index in range(0, len(input_features), batch_size):
            # 获取当前批次的输入和标签
            batch_input = input_features[batch_index: batch_index + batch_size]
            batch_labels = labels[batch_index: batch_index + batch_size]

            # 将数据转换为PyTorch张量
            xx = torch.tensor(batch_input, dtype=torch.float)
            yy = torch.tensor(batch_labels, dtype=torch.float).view(
                -1, 1
            )  # 将标签转换为列向量

            # 前向传播：通过神经网络得到预测结果
            prediction = my_nn(xx)

            # 计算损失
            loss = cost(prediction, yy)

            # 清零梯度，准备反向传播
            optimizer.zero_grad()

            # 反向传播，计算梯度
            loss.backward()

            # 更新参数
            optimizer.step()

            # 记录当前批次的损失
            batch_losses.append(loss.item())

        # 每100个迭代，打印平均损失
        if epoch % 100 == 0:
            avg_batch_loss = np.mean(batch_losses)
            losses.append(avg_batch_loss)
            print(f"{epoch} {avg_batch_loss}")

    # 将输入特征转换为torch张量，用于神经网络输入
    x = torch.tensor(input_features, dtype=torch.float)

    # 通过神经网络my_nn进行预测，并将结果转换为numpy数组
    predict = my_nn(x).data.numpy()

    return predict


if __name__ == "__main__":
    # 加载数据
    input_features, labels, feature_list = load_data("../data/temperature/temps.csv")
    # 构建网络模型,得到预测结果
    predict = build_model(input_features, labels)

    # 打印预测结果和真实值
    # print("预测结果：", predict)
    # print("真实值：", labels)

    # 用不同颜色的折线图绘制真实值和预测值，横坐标是样本的索引，纵坐标是温度值
    plt.plot(labels, "r", label="Actual")
    plt.plot(predict, "b", label="Predicted")
    plt.legend()
    plt.show()
