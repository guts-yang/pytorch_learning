"""
案例：
    演示张量的基础索引、切片、高级索引与 masked_select

理论说明：
    基础索引用整数和切片获取子张量
    高级索引可使用整型张量或布尔张量选择元素
    masked_select 返回一维张量，包含满足条件的元素

涉及到的函数：
    torch.masked_select()

需要掌握的函数：
    masked_select()

练习题：
    1. 取出二维张量的某一行与某一列
    2. 使用布尔张量选择所有大于阈值的元素
"""
import torch


def dm01():
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("x[0]:", x[0])
    print("x[:, 1]:", x[:, 1])
    print("x[0:2, 1:3]:", x[0:2, 1:3])
    print("=" * 50)


def dm02():
    x = torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    idx = torch.tensor([0, 2])
    print("x[idx]:", x[idx])
    print("-" * 50)
    mask = x > 45
    print("mask:", mask)
    print("masked_select:", torch.masked_select(x, mask))
    print("-" * 50)
    cols = torch.tensor([2, 0, 1])
    print("x[:, cols]:", x[:, cols])


if __name__ == "__main__":
    dm01()
    dm02()
