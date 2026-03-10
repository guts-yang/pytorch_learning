"""
深度学习激活函数可视化脚本
功能：可视化常用激活函数及其导数
作者：Claude
日期：2026-03-10
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（如果系统支持）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建x值范围
x = torch.linspace(-5, 5, 1000, requires_grad=True)

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + torch.exp(-x))

def tanh(x):
    """Tanh激活函数"""
    return torch.tanh(x)

def relu(x):
    """ReLU激活函数"""
    return F.relu(x)

def leaky_relu(x, negative_slope=0.1):
    """LeakyReLU激活函数"""
    return F.leaky_relu(x, negative_slope=negative_slope)

def elu(x, alpha=1.0):
    """ELU激活函数"""
    return F.elu(x, alpha=alpha)

def gelu(x):
    """GELU激活函数（GPT/BERT使用）"""
    return F.gelu(x)

def swish(x):
    """Swish激活函数（又称 SiLU）"""
    return x * sigmoid(x)

def softmax(x, dim=0):
    """Softmax激活函数"""
    return F.softmax(x, dim=dim)

def compute_derivative(func, x):
    """计算函数的导数"""
    y = func(x)
    # 计算梯度
    dy = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    return y.detach().numpy(), dy.detach().numpy()

# 定义激活函数及其参数
activations = [
    ('Sigmoid', sigmoid, {}),
    ('Tanh', tanh, {}),
    ('ReLU', relu, {}),
    ('LeakyReLU', leaky_relu, {'negative_slope': 0.1}),
    ('ELU', elu, {'alpha': 1.0}),
    ('GELU', gelu, {}),
    ('Swish', swish, {}),
    # ('Softmax', softmax, {}),  # Softmax通常用于多分类输出层，这里单独处理
]

# 创建画布
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
fig.suptitle('深度学习常用激活函数及其导数', fontsize=18, fontweight='bold')
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# 绘制每个激活函数
for idx, (name, func, params) in enumerate(activations):
    row = idx // 2
    col = (idx % 2) * 2

    # 计算函数值和导数
    y, dy = compute_derivative(lambda x: func(x, **params), x)

    # 绘制函数图像
    ax_func = axes[row, col]
    ax_func.plot(x.detach().numpy(), y, 'b-', linewidth=2, label=f'{name}(x)')
    ax_func.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_func.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_func.set_title(f'{name} 激活函数', fontsize=12, fontweight='bold')
    ax_func.set_xlabel('x', fontsize=10)
    ax_func.set_ylabel('f(x)', fontsize=10)
    ax_func.grid(True, alpha=0.3)
    ax_func.legend(loc='upper left', fontsize=9)
    ax_func.set_xlim(-5, 5)

    # 绘制导数图像
    ax_deriv = axes[row, col + 1]
    ax_deriv.plot(x.detach().numpy(), dy, 'r-', linewidth=2, label=f"{name}'(x)")
    ax_deriv.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_deriv.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_deriv.set_title(f'{name} 导数', fontsize=12, fontweight='bold')
    ax_deriv.set_xlabel('x', fontsize=10)
    ax_deriv.set_ylabel("f'(x)", fontsize=10)
    ax_deriv.grid(True, alpha=0.3)
    ax_deriv.legend(loc='upper left', fontsize=9)
    ax_deriv.set_xlim(-5, 5)

# 隐藏未使用的子图
for i in range(4):
    for j in range(4):
        if i * 2 + j >= len(activations) * 2:
            axes[i, j].set_visible(False)

plt.tight_layout()
plt.savefig('activation_functions.png', dpi=300, bbox_inches='tight')
print("激活函数图像已保存为 'activation_functions.png'")
plt.show()

# 单独绘制 Softmax
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig2.suptitle('Softmax 激活函数', fontsize=16, fontweight='bold')

# 示例输入
x_softmax = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y_softmax = softmax(x_softmax, dim=1)

# 绘制原始值
ax1.bar(['x1', 'x2', 'x3'], x_softmax[0].detach().numpy(), color='blue', alpha=0.7)
ax1.set_title('原始输入值', fontweight='bold')
ax1.set_ylabel('值')
ax1.set_ylim(0, 4)
ax1.grid(True, alpha=0.3, axis='y')

# 绘制 Softmax 输出
ax2.bar(['p1', 'p2', 'p3'], y_softmax[0].detach().numpy(), color='red', alpha=0.7)
ax2.set_title('Softmax 输出（概率）', fontweight='bold')
ax2.set_ylabel('概率')
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3, axis='y')
ax2.text(0.02, 0.95, f'总和: {y_softmax[0].sum().item():.4f}',
          transform=ax2.transAxes, fontsize=10)

plt.tight_layout()
plt.savefig('softmax_function.png', dpi=300, bbox_inches='tight')
print("Softmax 函数图像已保存为 'softmax_function.png'")
plt.show()

print("\n" + "="*60)
print("激活函数可视化完成！")
print("="*60)
print("\n各激活函数特点总结：")
print("- Sigmoid: 输出范围(0,1)，适合二分类，容易梯度消失")
print("- Tanh: 输出范围(-1,1)，零中心化，适合隐藏层")
print("- ReLU: 简单高效，解决梯度消失，但有死神经元问题")
print("- LeakyReLU: ReLU改进，负区间有小斜率，缓解死神经元")
print("- ELU: 负区间指数衰减，输出接近零，但计算较慢")
print("- GELU: 平滑激活函数，GPT/BERT等Transformer模型使用")
print("- Swish: 自门控激活，x*sigmoid(x)，在某些任务表现优异")
print("- Softmax: 多分类输出层，将输出转换为概率分布")
