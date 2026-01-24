"""
案例：
    演示PyTorch中如何创建线性和随机张量

涉及到的函数：
    torch.arange() 和 torch.linspace() 创建线性张量
    torch.random.initial_seed() 和 torch.manual_seed() 设置随机种子
    torch.rand() 和 torch.randn() 创建随机张量
    torch.randint(low,  high, size=()) 创建随机整数张量

需要掌握的函数：
    arange(), linspace(), manual_seed(), randint()
"""
import torch
import numpy as np
# 1. 定义函数：创建线性张量
def dm01():
    # 场景1 arrange(start, end, step)步长
    # 参数1：起始值；参数2：结束值；参数3：步长
    t1 = torch.arange(0, 10, 2)
    print("t1:", t1)
    print("-" * 50)
    # 场景2 linspace(start, end, steps)等差数列，总步数
    # 参数1：起始值；参数2：结束值；参数3：总步数
    t2 = torch.linspace(1, 10, 6)
    print("t2:", t2)
# 2. 定义函数，创建随机函数
def  dm02():
    print("=" * 50)
    # 设置随机种子
    # torch.initial_seed() # 默认采用当前系统的时间戳作为随机种子
    # print("t1:", t1)
    torch.manual_seed(10) # 设置随机种子
    # 场景1：创建指定形状的随机张量
    t1 = torch.rand(size=(2, 3))
    print("t1:",{t1})
    print("-" * 50)
    # 场景2：符合正态分布的随机张量
    t2 = torch.randn(size=(2, 3))
    print("t2:",{t2})
    print("-" * 50)
    # 场景3：创建随机整数张量
    # 参数1：起始值；参数2：结束值；参数3：张量形状
    t3 = torch.randint(low=0, high=10, size=(2, 3))
    print("t3:", {t3})
    print("-" * 50)
#  3. 测试函数
if __name__ == '__main__':
    dm01()
    dm02()