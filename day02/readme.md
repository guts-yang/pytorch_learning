# Day02 学习笔记汇总

## 目录结构
- 01张量与Numpy之间的相互转换.py
- 02张量的基本运算.py
- 03张量点乘和矩阵乘法.py
- 04张量的常用运算函数.py
- 05张量的索引操作.py
- 06张量的形状操作reshape函数.py
- 07张量的形状操作unsqueeze.py
- 08张量的形状操作transpose.py
- 09张量的形状操作view.py
- 10张量的拼接操作.py
- 11自动微分模块.py
- 12自动微分模块案例-更新一次参数.py
- 13自动微分模块案例-循环更新参数.py

## 笔记分类说明
- 数据互转：张量与 NumPy 之间的共享内存转换
- 基本运算：逐元素、比较、幂运算及对应函数
- 线性代数：点乘、矩阵乘法与批量矩阵乘法
- 常用函数：聚合函数与数学函数的典型用法
- 索引与切片：基础索引、高级索引与掩码选择
- 形状操作：reshape、unsqueeze、transpose、view 与拼接
- 自动微分：计算图与训练循环的完整流程

## 关键知识点摘要
- from_numpy 与 numpy 的共享内存机制与解除共享方式
- add、sub、mul、div、pow 与运算符的对应关系
- dot、mv、mm、matmul 与 @ 的使用边界
- sum、mean、max、min 与 exp、log 的常见模式
- 布尔索引与 masked_select 的一维返回特性
- view 需要连续内存，reshape 更灵活
- unsqueeze 用于维度对齐与广播准备
- cat 与 stack 的维度差异
- requires_grad、grad_fn 与 backward 的基本原理

## 相关资源
- https://pytorch.org/docs/stable/tensors.html
- https://pytorch.org/docs/stable/torch.html
- https://pytorch.org/docs/stable/autograd.html
