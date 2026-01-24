"""
案例：
    演示 reshape 的用法与返回视图的条件

理论说明：
    reshape 会尽量返回视图，当内存不连续时可能返回拷贝
    reshape 与 view 的差异在于 reshape 更具适配性
    使用 is_contiguous 可判断内存是否连续

涉及到的函数：
    Tensor.reshape(), Tensor.is_contiguous(), Tensor.view()

需要掌握的函数：
    reshape()

练习题：
    1. 对连续张量进行 reshape 并验证共享存储
    2. 对转置后的张量 reshape 并观察是否触发拷贝
"""
import torch


def dm01():
    x = torch.arange(12).reshape(3, 4)
    y = x.reshape(2, 6)
    print("x:", x)
    print("y:", y)
    print("x is_contiguous:", x.is_contiguous())
    print("y is_contiguous:", y.is_contiguous())
    print("=" * 50)


def dm02():
    a = torch.arange(12).reshape(3, 4)
    b = a.transpose(0, 1)
    c = b.reshape(2, 6)
    print("b:", b)
    print("b is_contiguous:", b.is_contiguous())
    print("c:", c)
    print("c is_contiguous:", c.is_contiguous())


if __name__ == "__main__":
    dm01()
    dm02()
