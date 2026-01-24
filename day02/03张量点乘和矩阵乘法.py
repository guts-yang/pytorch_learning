"""
案例：
    演示张量点乘与矩阵乘法的差异

理论说明：
    torch.dot 只适用于一维张量，计算向量点乘
    torch.mv 适用于二维矩阵与一维向量
    torch.mm 适用于二维矩阵与二维矩阵
    torch.matmul 和 @ 支持更一般的矩阵乘法与批量矩阵乘法

涉及到的函数：
    torch.dot(), torch.mv(), torch.mm(), torch.matmul()

需要掌握的函数：
    dot(), mv(), mm(), matmul()

练习题：
    1. 构造不同维度的张量，分别尝试 dot/mv/mm/matmul
    2. 使用 @ 运算符验证与 matmul 一致的结果
"""
import torch


def dm01():
    v1 = torch.tensor([1.0, 2.0, 3.0])
    v2 = torch.tensor([4.0, 5.0, 6.0])
    print("dot:", torch.dot(v1, v2))
    print("=" * 50)


def dm02():
    m = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    v = torch.tensor([1.0, 1.0, 1.0])
    print("mv:", torch.mv(m, v))
    print("-" * 50)
    n = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    print("mm:", torch.mm(m, n))
    print("-" * 50)
    a = torch.randn(2, 3, 4)
    b = torch.randn(2, 4, 5)
    print("matmul:", torch.matmul(a, b).shape)
    print("operator @:", (a @ b).shape)


if __name__ == "__main__":
    dm01()
    dm02()
