import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

def create_dataset():
    x = torch.randn(100, 1)
    y = 3 * x + 2 + 0.1 * torch.randn(100, 1)   # 参数分别为3，2，高斯噪声的标准差为0.1
    return x, y

def train(x, y, coef):
    # 1. 创建数据集，把tensor->数据集对象->数据加载器对象
    dataset = TensorDataset(x, y)
    # 2. 创建数据加载器对象
    # 参数1：数据集，参数2：批量大小，参数3：是否打乱数据（训练集打乱，测试集不打乱）
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    # 3. 创建线性模型
    # 参数1：输入特征维度，参数2：输出特征维度
    modal = nn.Linear(1, 1)
    # 4. 定义损失函数对象
    # 定义损失函数，参数1：损失函数类型
    criterion = nn.MSELoss()
    # 5. 定义优化器对象
    # 定义优化器，参数1：模型参数，参数2：学习率
    optimizer = torch.optim.SGD(modal.parameters(), lr=0.01)
    # 6. 训练模型
    epochs, loss_list, total_loss, total_sample = 100, [], 0, 0
    for epoch in range(epochs): # 训练轮数:0, 1, 2, ..., 99
        for train_x, train_y in dataloader: # 训练7批：16，16，16，16，16，16，4
            # 6.1 前向传播，记录：模型的预测结果
            z = modal(train_x) # 模型的预测结果
            # 6.2 计算（每轮平均）损失，记录：损失函数的计算过程
            loss = criterion(z, train_y.reshape(-1, 1)) # -1 自动计算维度
            # 6.3 计算总损失，和样本批次数
            total_loss += loss.item()# loss.item() 获取标量值，train_x.size(0) 获取当前批次的样本数量
            total_sample += train_x.size(0) # 累加样本数量
            # 6.4 反向传播，记录：计算损失函数对模型参数的导数，即梯度
            optimizer.zero_grad() # 梯度清零，可以放在反向传播之前或之后，效果一样
            loss.backward()
            optimizer.step() # 更新模型参数
        # 6.5 计算平均损失，记录：每轮的平均损失
        loss_list.append(total_loss / total_sample)
        print(f"epoch轮:{epoch+1}, loss平均损失:{loss_list[-1]:.4f}")
    # 7. 输出模型参数，记录：训练完成后的模型参数
    print(f'epochs:{epochs}, loss_list:{loss_list}')
    print("模型参数:", list(modal.parameters()))
    # 8. 画图，记录：损失曲线
    fname(loss_list)

# 画图
def fname(arg):
    import matplotlib.pyplot as plt
    # 参数1：x轴数据，参数2：y轴数据，参数3：标签
    plt.plot(arg)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss curve")
    plt.show()

if __name__ == "__main__":
    x, y = create_dataset()
    # 记录：模型参数的初始值
    # coef：模型参数的初始值，参数1：初始值，参数2：是否需要求梯度，参数3：数据类型(只有浮点型才支持求梯度)
    coef = torch.tensor([0.0], requires_grad=True)
    train(x, y, coef)