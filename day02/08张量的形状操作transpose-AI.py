"""
案例：
    演示 transpose 交换维度

理论说明：
    transpose 仅交换两个维度 等价于 转置
    permute 可以一次性重排多个维度

涉及到的函数：
    Tensor.transpose(), Tensor.permute()

需要掌握的函数：
    transpose(), permute()

练习题：
    1. 对二维张量交换行列并对比结果
    2. 对三维张量交换 dim0 与 dim2
"""
import torch


def dm01():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print("x:", x)
    print("transpose(0,1):", x.transpose(0, 1))
    print("=" * 50)


def dm02():
    y = torch.arange(24).reshape(2, 3, 4)
    print("y shape:", y.shape)
    z = y.transpose(0, 2)
    print("z shape:", z.shape)
    print("permute:", y.permute(2, 0, 1).shape)


if __name__ == "__main__":
    dm01()
    dm02()
