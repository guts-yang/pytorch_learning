import torch
import numpy as np
"""
张量与Numpy之间的相互转换
案例：
    1. 张量转换为Numpy数组
    2. Numpy数组转换为张量
理论说明：
    1. 张量转换为Numpy数组：使用张量的numpy()方法即可将张量转换为Numpy数组。
    2. Numpy数组转换为张量：使用torch.from_numpy()方法即可将Numpy数组转换为张量。
浅拷贝：
    1. 张量转换为Numpy数组：使用张量的numpy()方法即可将张量转换为Numpy数组。
    2. Numpy数组转换为张量：使用torch.from_numpy()方法即可将Numpy数组转换为张量。
    3. 注意：浅拷贝的张量与Numpy数组共享内存，对其中一个的修改会影响到另一个。
涉及到的API：
    场景1：张量 -> numpy  nd数组对象
        张量对象.detach().numpy()        共享内存
        张量对象.detach().numpy().copy() 不共享内存，链式编程写法
    场景2：numpy -> 张量  张量对象
        torch.from_numpy(nd数组对象)  共享内存
        torch.tensor(nd数组对象)      不共享内存，链式编程写法
    场景3：从标量张量中 提取内容
        标量张量对象.item()  提取标量张量中的内容，返回Python标量
掌握：
    张量 -> numpy: 张量对象.detach().numpy()      detach()方法将张量从计算图中分离出来，避免梯度计算
    numpy -> 张量: torch.from_numpy(nd数组对象)  共享内存
    标量张量 ->  Python标量: 标量张量对象.item()
"""
def dem01():
    """
    张量转换为Numpy数组
    """
    t1 = torch.tensor([1,2,3,4,5,6,7])
    print(f"t1:{t1}","dtype:",t1.dtype)
    n1 = t1.detach().numpy()
    print(n1)
    print("-"*50)
    t2 = torch.tensor([1,2,3])
    print(t2)
    n2 = t2.detach().numpy().copy()
    print(n2)
    print("-"*50)
    n2[0] = 100
    print(n2)
    print(t2,"copy()深拷贝不会共享存储空间")
    print("="*50)

def dem02():
    """
    Numpy数组转换为张量
    """
    # 1. 创建数组
    n1 = np.array([1,2,3,4,5,6,7])
    print(n1)
    print("-"*50)
    # 2. 从数组创建张量
    t1 = torch.from_numpy(n1)
    t2 = torch.tensor(n1)
    print(f"t1:{t1}","dtype:",t1.dtype)# 1，2，3，4，5，6，7
    print(f"t2:{t2}","dtype:",t2.dtype)# 1，2，3，4，5，6，7
    print("-"*50)
    # 3. 演示上述方式   是否共享内存
    n1[0] = 100
    print(n1)# 100，2，3，4，5，6，7
    print(t1,"from_numpy()浅拷贝会共享存储空间")# 100，2，3，4，5，6，7
    print(t2,"tensor()深拷贝不会共享存储空间")# 1，2，3，4，5，6，7
    print("="*50)

def dem03():
    """
    标量张量 ->  Python标量: 标量张量对象.item()
    从标量张量中提取内容，返回Python标量
    """
    t1 = torch.tensor(30)
    n1 = t1.item()
    print(f"t1:{t1}","item()方法提取标量张量中的内容，返回Python标量")# 30
    print(n1, "dtype:",type(n1))# 30 <class 'int'>
    print("="*50)
    
if __name__ == '__main__':
    #dem01()
    dem02()
    dem03()