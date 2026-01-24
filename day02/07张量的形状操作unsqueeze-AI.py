"""
案例：
    演示 unsqueeze 增加维度的作用与典型应用

理论说明：
    unsqueeze 在指定维度插入大小为 1 的维度
    常用于广播准备或与批量维度对齐

涉及到的函数：
    Tensor.unsqueeze()

需要掌握的函数：
    unsqueeze()

练习题：
    1. 将一维张量分别在 dim=0 与 dim=1 上扩维
    2. 使用 unsqueeze 让两个张量实现广播相加
"""
import torch


def dm01():
    x = torch.tensor([1, 2, 3])
    print("x shape:", x.shape)
    print("unsqueeze(0):", x.unsqueeze(0).shape)
    print("unsqueeze(1):", x.unsqueeze(1).shape)
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
