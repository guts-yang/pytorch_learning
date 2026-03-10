"""
案例：
    演示 view 的使用限制与错误排查

理论说明：
    view 只适用于连续内存的张量; 使用is_contiguous()检查张量是否连续
    若张量不连续，需要先调用 contiguous()
    不连续张量使用 view 会抛出异常

涉及到的函数：
    Tensor.view(), Tensor.is_contiguous(), Tensor.contiguous()

需要掌握的函数：
    view()

练习题：
    1. 对连续张量使用 view 并验证形状变化
    2. 对转置张量尝试 view 并处理异常
"""
import torch


def dm01():
    x = torch.arange(12)
    print("x:", x)
    y = x.view(3, 4)
    print("y:", y)
    print("y shape:", y.shape)
    print("y is_contiguous:", y.is_contiguous())
    print("=" * 50)


def dm02():
    a = torch.arange(12).reshape(3, 4)
    b = a.transpose(0, 1)   # 转置后变为4行3列
    print("b is_contiguous:", b.is_contiguous())
    try: # 尝试对不连续张量使用 view 会抛出异常
        c = b.view(2, 6)
        print("c:", c)
    except RuntimeError as e:
        print("view error:", str(e))
    c2 = b.contiguous().view(2, 6)
    print("c2:", c2)


if __name__ == "__main__":
    dm01()
    dm02()
