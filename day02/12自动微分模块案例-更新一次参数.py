"""
案例：
    使用自动微分更新一次参数

理论说明：
    前向传播得到预测值
    通过损失函数构建标量损失
    backward 计算梯度
    通过优化器执行一次参数更新

涉及到的函数：
    torch.nn.Linear, torch.nn.MSELoss, torch.optim.SGD, Tensor.backward()

需要掌握的函数：
    backward(), step()

练习题：
    1. 将线性层输入维度改为 2，验证输出形状变化
    2. 手动实现一次梯度更新并对比优化器结果
"""
import torch


def dm01():
    torch.manual_seed(0)
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])
    model = torch.nn.Linear(1, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    pred = model(x)
    loss = criterion(pred, y)
    print("loss before:", loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    pred2 = model(x)
    loss2 = criterion(pred2, y)
    print("loss after:", loss2.item())


if __name__ == "__main__":
    dm01()
