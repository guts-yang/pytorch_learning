"""
案例：
    演示张量的逐元素运算与比较运算

理论说明：
    张量的逐元素运算遵循广播规则
    加减乘除与幂运算可使用 Python 运算符或对应的 torch 函数
    比较运算返回布尔张量，可用于后续索引或统计

涉及到的函数：
    torch.add(), torch.sub(), torch.mul(), torch.div(), torch.pow()
    torch.eq(), torch.ne(), torch.gt(), torch.ge(), torch.lt(), torch.le()

需要掌握的函数：
    add(), sub(), mul(), div(), pow()

练习题：
    1. 使用运算符与函数分别实现逐元素加减乘除并对比结果
    2. 构造比较运算并统计 True 的数量
"""
import torch


def dm01():
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    b = torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.float32)
    print("a + b:", a + b)
    print("a - b:", a - b)
    print("a * b:", a * b)
    print("a / b:", a / b)
    print("a ** 2:", a ** 2)
    print("-" * 50)
    print("add:", torch.add(a, b))
    print("sub:", torch.sub(a, b))
    print("mul:", torch.mul(a, b))
    print("div:", torch.div(a, b))
    print("pow:", torch.pow(a, 2))
    print("=" * 50)


def dm02():
    x = torch.tensor([1, 2, 3, 4])
    y = torch.tensor([2, 2, 1, 4])
    print("x == y:", x == y)
    print("x != y:", x != y)
    print("x > y:", x > y)
    print("x >= y:", x >= y)
    print("x < y:", x < y)
    print("x <= y:", x <= y)
    print("count_true:", (x >= y).sum())


if __name__ == "__main__":
    dm01()
    dm02()
