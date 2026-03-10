"""
案例：
    演示自动微分，循环实现
需求：
    求 loss = w ** 2 + 20 的极小值点并打印loss是最小值时的梯度
"""
import torch
def dm01():
    w = torch.tensor([10.0], requires_grad=True) # 需要求梯度
    print("初始w:", w) # 初始w: tensor([10.], requires_grad=True)
    for epoch in range(1, 201):
        # 1. 正向计算（前向传播）
        loss = w ** 2  + 20
        loss.backward() # 计算梯度
        with torch.no_grad(): # 禁止梯度计算
            w -= 0.01 * w.grad # 更新参数  
            print(f"epoch: {epoch}, 权重: {w.item():.4f}, 结果: {loss.item():.4f}, (w - 0.01 * w.grad): {w - 0.01 * w.grad},梯度: {w.grad.item():.4f}")
            w.grad.zero_() # 清零梯度，如果没有写这行代码，梯度会累积

if __name__ == "__main__":
    dm01()