"""
案例：
    演示张量的聚合函数与数学函数

理论说明：
    聚合函数可在整体或指定维度上计算统计量
    数学函数对张量逐元素计算，支持广播

涉及到的函数：
    torch.sum(), torch.mean(), torch.max(), torch.min()
    torch.exp(), torch.log()

需要掌握的函数：
    sum(), mean(), max(), min(), exp(), log()

练习题：
    1. 分别在 dim=0 与 dim=1 上计算 sum 与 mean
    2. 构造正数张量，验证 exp 与 log 的互逆关系
"""
import torch


def dm01():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print("sum:", torch.sum(x))
    print("sum dim0:", torch.sum(x, dim=0))
    print("sum dim1:", torch.sum(x, dim=1))
    print("-" * 50)
    print("mean:", torch.mean(x))
    print("mean dim0:", torch.mean(x, dim=0))
    print("mean dim1:", torch.mean(x, dim=1))
    print("-" * 50)
    print("max:", torch.max(x))
    print("min:", torch.min(x))
    print("max dim0:", torch.max(x, dim=0))
    print("min dim1:", torch.min(x, dim=1))
    print("=" * 50)


def dm02():
    y = torch.tensor([0.5, 1.0, 2.0])
    exp_y = torch.exp(y)
    log_exp_y = torch.log(exp_y)
    print("y:", y)
    print("exp:", exp_y)
    print("log(exp):", log_exp_y)


if __name__ == "__main__":
    dm01()
    dm02()
