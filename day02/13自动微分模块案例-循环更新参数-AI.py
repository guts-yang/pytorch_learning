"""
案例：
    使用自动微分进行循环训练

理论说明：
    训练循环包含前向传播、损失计算、反向传播、梯度清零与参数更新
    梯度会累积，因此每次更新前需要清零

涉及到的函数：
    torch.nn.Linear, torch.nn.MSELoss, torch.optim.SGD, optimizer.zero_grad()

需要掌握的函数：
    zero_grad(), backward(), step()

练习题：
    1. 修改学习率并观察 loss 的变化趋势
    2. 增加训练轮数并记录最后一次 loss
"""
import torch


def dm01():
    torch.manual_seed(0)
    x = torch.linspace(0, 3, 20).unsqueeze(1)
    y = 2 * x + 1
    model = torch.nn.Linear(1, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(1, 21):
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print("epoch:", epoch, "loss:", loss.item())


if __name__ == "__main__":
    dm01()
