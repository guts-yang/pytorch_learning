"""
案例：
    演示PyTorch中如何创建全0全1指定值张量
涉及到的函数如下：
    torch.zeros() 和 torch.ones() 创建全0全1指定值张量
    torch.full() 创建指定值张量
    torch.empty() 创建空张量
需要掌握的函数：
    zeros() full()
"""
import torch
# torch.zeros() 和 torch.ones() 创建全0全1指定值张量
def dem01():
    print("dem01")
    # 场景1：创建全1张量，形状为[2, 4]
    # 参数：shape 张量的形状，例如[2, 4]表示2行4列的张量
    t1 = torch.ones([2, 4])
    print("t1:", t1)
    print("-" * 50)
    # 场景2：创建全0张量，形状为[2, 4]
    # 参数：shape 张量的形状，例如[2, 4]表示2行4列的张量
    t2 = torch.zeros([2, 4])
    print("t2:", t2)
    print("-" * 50)
    # 场景3：创建指定值张量，形状为[2, 4]，值为5
    # 参数1：shape 张量的形状，例如[2, 4]表示2行4列的张量
    # 参数2：fill_value 张量)的值，例如5表示张量的值为5
    t3 = torch.full([2, 4], 5)
    print("t3:", t3)
    print("-" * 50)
    # 场景4：创建[2, 4]张量
    t4 = torch.tensor([[1, 2], [3, 4]])
    print("t4:", t4)
    print("-" * 50)
    # 场景5：创建与t4形状相同的全1张量
    t5 = torch.ones_like(t4)
    print("t5:", t5)
    print("-" * 50)
def dem02():
    print("dem02")
    # 创建指定值张量，形状为[2, 4]，值为5
    five = torch.full([2, 4], 5)
    print(five)
    # 创建空张量，形状为[2, 4]
    empty = torch.empty([2, 4])
    print(empty)
# torch.full() 创建指定值张量
# torch.empty() 创建空张量

if __name__ == '__main__':
    dem01()
    dem02()
