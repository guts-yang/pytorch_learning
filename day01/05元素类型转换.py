"""
案例：
    创建指定类型的张量
涉及到的函数如下：
    type(torch支持的数据类型)
    half()/float()/double()/int()/long()
需要掌握的函数：
    type(torch.数据类型)：将已有张量转换为指定类型
"""
import torch
# 场景1：直接创建指定类型的张量
t1 = torch.tensor([1,2.,3.],dtype=torch.float32) # 默认为float32
print(f"t1={t1},dtype={t1.dtype}")
print("-"*50)
# 场景2：将已有张量转换为指定类型
# 思路1：type()函数，推荐掌握
t2 = t1.type(torch.int16)
print(f"t2={t2},dtype={t2.dtype}")
print("="*50)
# 思路2：half()/float()/double()/int()/long()
print(t2.half())    # float16
print(t2.float())   # float32
print(t2.double())  # float64
print(t2.short())   # int16
print(t2.int())     # int32
print(t2.long())    # int64


