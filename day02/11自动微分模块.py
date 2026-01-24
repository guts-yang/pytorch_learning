"""
案例：
    演示自动微分模块的基本概念

理论说明：
    requires_grad=True 表示需要对该张量求梯度
    grad_fn 记录张量的计算图来源
    backward 会沿计算图反向传播并累积梯度

涉及到的函数：
    Tensor.backward(), Tensor.grad, Tensor.grad_fn

需要掌握的函数：
    backward()

练习题：
    1. 构造一个简单计算图并打印 grad_fn
    2. 对标量损失调用 backward 并查看梯度
"""
import torch


def dm01():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2
    z = y.sum()
    print("y grad_fn:", y.grad_fn)
    print("z grad_fn:", z.grad_fn)
    z.backward()
    print("x.grad:", x.grad)


if __name__ == "__main__":
    dm01()
