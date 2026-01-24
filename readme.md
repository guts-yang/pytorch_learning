# PyTorch 学习笔记总览

## 目录结构
- day01/
  - 01框架简介.py
  - 02张量基本创建方式.py
  - 03创建线性和随机张量.py
  - 04创建全0全1指定值张量.py
  - 05元素类型转换.py
  - readme.md
- day02/
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
  - readme.md

## 笔记分类说明
- Day01：张量基础与创建方式、初始化与类型转换
- Day02：张量运算、索引、形状操作与自动微分训练流程

## 关键知识点摘要
- 张量是 PyTorch 的核心数据结构，覆盖创建、初始化与类型变换
- 张量与 NumPy 共享内存互转，需注意拷贝与视图
- 基本运算、比较运算与矩阵乘法的使用场景与接口差异
- reshape、view、unsqueeze、transpose 与拼接的维度操作规律
- 自动微分的计算图、梯度计算与训练循环范式

## 相关资源
- https://pytorch.org/docs/stable/index.html
- https://pytorch.org/docs/stable/tensors.html
- https://pytorch.org/docs/stable/autograd.html
