"""
案例：
    演示 cat 与 stack 的区别

理论说明：
    torch.cat 在已有维度上拼接，维度数量不变
    torch.stack 在新维度上堆叠，维度数量增加
    dim 参数决定拼接或堆叠的轴

涉及到的函数：
    torch.cat(), torch.stack()

需要掌握的函数：
    cat(), stack()

练习题：
    1. 对两个二维张量分别在 dim=0 与 dim=1 上 cat
    2. 对多个一维张量 stack 并观察维度变化
"""
import torch


def dm01():
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])
    print("cat dim0:", torch.cat([a, b], dim=0))
    print("cat dim1:", torch.cat([a, b], dim=1))
    print("=" * 50)


def dm02():
    v1 = torch.tensor([1, 2, 3])
    v2 = torch.tensor([4, 5, 6])
    v3 = torch.tensor([7, 8, 9])
    print("stack dim0:", torch.stack([v1, v2, v3], dim=0))
    print("stack dim1:", torch.stack([v1, v2, v3], dim=1))


if __name__ == "__main__":
    dm01()
    dm02()
