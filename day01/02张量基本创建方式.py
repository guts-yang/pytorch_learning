import torch
import numpy as np
# 1. 使用torch.tensor()创建标量张量
def dm01():
    # 1. 使用torch.tensor()创建标量张量
    t1 = torch.tensor(10)
    print(f"t1 = {t1}, type = {t1.type()}")
    print('-' * 50)
    # 2. 使用torch.tensor()创建向量张量
    data = [1, 2, 3]
    t2 = torch.tensor(data)
    print(f"t2 = {t2}, type = {t2.type()}")
    print('-' * 50)
    # 3. 创建二维张量
    data = [[1, 2, 3], [4, 5, 6]]
    t3 = torch.tensor(data)
    print(f"t3 = {t3}, type = {t3.type()}")
    print('-' * 50)
    # 4. 生成一个2x3的随机整数矩阵，元素范围为[0, 10)
    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.tensor(data)
    print(f"t3 = {t3}, type = {t3.type()}")
    # print('-' * 50)
    # # 4. 尝试直接创建张量
    # try:
    #     t4 = torch.tensor(2, 3)
    #     print(f"t4 = {t4}, type = {t4.type()}")
    # finally:
    #     print("try-finally语句执行完成")
    
    print('=' * 50)
# 2. 演示torch.Tensor()函数的基本用法
def dm02():
    # 注意：torch.Tensor()函数默认创建的是torch.FloatTensor类型的张量
    t1 = torch.Tensor([1, 2, 3])
    print(f"t1 = {t1}, type = {t1.type()}")
    print('-' * 50)
    # 2. Tensor()方式可以基于形状直接创建张量
    # (2, 3)的意思是2行3列的矩阵，每个元素都是0
    t2 = torch.Tensor(2, 3)
    print(f"t2 = {t2}, type = {t2.type()}")
    print('-' * 50)
    # 3. 创建3维张量（例如，批量大小为2，每个样本有3个通道，每个通道有4行5列）
    t3 = torch.Tensor(2, 3, 4, 5)
    print(f"t3 = {t3}, type = {t3.type()}")

if __name__ == "__main__":
# pass意思是占位符，不执行任何操作
    dm01()
    dm02()
    pass