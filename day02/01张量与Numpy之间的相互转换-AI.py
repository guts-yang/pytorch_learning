"""
案例：
    演示张量与NumPy数组之间的相互转换

理论说明：
    torch.from_numpy() 可以把 NumPy 数组转换为张量，二者共享同一块内存
    Tensor.numpy() 可以把张量转换为 NumPy 数组，二者共享同一块内存
    共享内存意味着任意一方修改数据，另一方会同步变化
    若需要解除共享，可使用 clone() 或 copy()

涉及到的函数：
    torch.from_numpy(), Tensor.numpy(), Tensor.clone(), ndarray.copy()

需要掌握的函数：
    from_numpy(), numpy()

练习题：
    1. 创建一个二维 NumPy 数组并转换为张量，验证共享内存的变化
    2. 使用 clone() 或 copy() 解除共享并验证数据不再联动
"""
import torch
import numpy as np


def dm01():
    np_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    t1 = torch.from_numpy(np_data)
    # t1和np_data共享内存空间
    print("np_data:", np_data)
    print("t1:", t1)
    print("-" * 50)
    np_data[0, 0] = 100
    # 验证共享内存，修改 np_data 会同步到 t1
    print("np_data:", np_data)
    print("t1:", t1)
    print("-" * 50)
    t1[1, 2] = 200
    # 验证共享内存，修改 t1 会同步到 np_data
    print("np_data:", np_data)
    print("t1:", t1)
    print("=" * 50)


def dm02():
    t2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    np_view = t2.numpy()
    print("t2:", t2)
    print("np_view:", np_view)
    print("-" * 50)
    np_view[0, 1] = 99
    print("t2:", t2)
    print("np_view:", np_view)
    print("-" * 50)
    t3 = t2.clone()
    np_copy = t2.numpy().copy()
    t3[0, 0] = -1
    np_copy[1, 1] = -2
    print("t3:", t3)
    print("np_copy:", np_copy)
    print("t2:", t2)


if __name__ == "__main__":
    dm01()
    dm02()
