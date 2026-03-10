import torch
import numpy as np
"""
案例：
    演示张量的逐元素运算与比较运算
涉及到的API：
    逐元素运算：
        add(), sub(), mul(), div(), pow(), neg()   ->加减乘除幂、取负
        add_(), sub_(), mul_(), div_(), pow_()   ->功能同上，不过可以修改源数据，类似Pandas部分得到inplace = True
    比较运算：
        eq(), ne(), gt(), ge(), lt(), le()  ->等于、不等于、大于、大于等于、小于、小于等于
掌握：
    + - * / **  对应 add(), sub(), mul(), div(), pow()
"""
# 1. 创建张量
t1 = torch.tensor([1,2,3])
print( f"t1 = {t1}" )
print("-"*50)
# 2. 演示：加法
t2 = t1.add(10) # 不修改源数据
t2 += 10        # 同上
# t2 = t1 + 10  # 同上
print("t1.add(10), t2 += 10不修改源数据：")
print( f"t1 = {t1}" )
print( f"t2 = {t2}" )
print("-"*50)
print( f"t1 = {t1}" )
t2 = t1.add_(10) # 修改源数据
print("t1.add_(10)修改源数据：")
print( f"t1 = {t1}" )
print( f"t2 = {t2}" )
print("="*50)

# 3. 演示：减法
print(f"t1 = {t1}","dtype =", t1.dtype)
t3 = t1.sub(2) # 不修改源数据
print(f"t3 = {t3}","dtype =", t3.dtype)
# print("-"*50)
t3 -= 2        # 同上
# t3 = t1 - 2  # 同上
print(f"t3 = {t3}")