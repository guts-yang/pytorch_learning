"""
案例：
    演示 unsqueeze 增加维度的作用与典型应用
    演示 squeeze

理论说明：
    unsqueeze 在指定维度插入大小为 1 的维度
    常用于广播准备或与批量维度对齐
    squeeze 用于移除大小为 1 的维度 
涉及到的函数：
    Tensor.unsqueeze(), Tensor.squeeze()

需要掌握的函数：
    unsqueeze(), squeeze()

练习题：
    1. 将一维张量分别在 dim=0 与 dim=1 上扩维
    2. 使用 unsqueeze 让两个张量实现广播相加
"""
import torch


def dm01():
    x = torch.tensor([1, 2, 3])
    print("x shape:", x.shape)
    print("unsqueeze(0):", x.unsqueeze(0).unsqueeze(0).shape) # 在第0维插入一个大小为1的维度，变成(1, 3)
    print("unsqueeze(1):", x.unsqueeze(1).shape) # 在第1维插入一个大小为1的维度，变成(3, 1)
    print("squeeze(1):", x.unsqueeze(1).squeeze(1).shape) # 先在第1维插入一个大小为1的维度，再移除第1维，回到(3,)
    print("=" * 50)


def dm02():
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([[10], [20], [30]])
    a2 = a.unsqueeze(0)
    print("a2 shape:", a2.shape)
    print("b shape:", b.shape)
    print("broadcast add:", a2 + b)


if __name__ == "__main__":
    dm01()
    dm02()
