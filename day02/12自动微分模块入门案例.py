"""
案例：
自动微分模块入门案例
回馈：
    权重更新公式：
    w新 = w旧 - 学习率 * 梯度
    梯度 = loss 对 w 的导数
理论说明：
    自动微分模块通过记录操作历史来计算梯度
"""
import torch 
# 1. 定义变量，记录：初始权重w旧
# 参数1：初始值，参数2：是否需要求梯度，参数3：数据类型(只有浮点型才支持求梯度)
w = torch.tensor([10], requires_grad=True, dtype=torch.float32) # 需要求梯度
# 2. 定义损失函数，记录：损失函数的计算过程
loss =  2 * w ** 2 + 20 
print(f"梯度函数类型：{loss.grad_fn}") # 梯度函数类型：<PowBackward0 object at 0x0000021B8C9F3A30>
print("loss:", loss.sum()) # loss: 200.0
# 3. 反向传播，记录：计算 loss 对 w 的导数，即梯度
loss.sum().backward() # loss.sum().backward() 会沿着计算图反向传播，计算出 loss 对 w 的导数，并将结果存储在 w.grad 中
print("w.grad:", w.grad) # w.grad: 40.0
print("=" * 50)
# 4. 更新权重，记录：根据梯度更新权重       
a = 0.01 # 学习率
w_new = w.data - a * w.grad # w新 = w旧 - 学习率 * 梯度
print("初始权重w:", w) # w: tensor([10.], requires_grad=True)
print("w_new:", w_new) # w_new: tensor([6.], grad_fn

