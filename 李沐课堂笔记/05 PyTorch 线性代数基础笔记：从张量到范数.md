# PyTorch 线性代数基础笔记：从张量到范数

# PyTorch 线性代数基础笔记

---

## 一、张量与矩阵基础

### 1. 张量创建

```Python

import torch

# 一维张量（向量）
x = torch.arange(4)  # 生成 [0, 1, 2, 3]

# 二维张量（矩阵）：5行4列
A = torch.arange(20).reshape(5, 4)
# 结果：
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11],
#         [12, 13, 14, 15],
#         [16, 17, 18, 19]])
```

### 2. 矩阵转置

```Python

A.T  # 矩阵转置，行变列、列变行
# 结果形状：(4, 5)
```

### 3. 张量克隆（深拷贝）

```Python

B = A.clone()  # 分配新内存，创建A的副本，修改B不会影响A
```

---

## 二、按元素运算（Hadamard积）

两个矩阵的**按元素乘法**称为 Hadamard 积（数学符号： $\odot$ ），PyTorch 中直接用 `*` 实现：

```Python

# 若 A 和 B 形状相同，则 A * B 为 Hadamard 积
C = A * B  # 对应位置元素相乘
```

---

## 三、求和与均值

### 1. 按轴求和

```Python

# 按轴0（列方向）求和，输出形状丢失轴0维度 → (4,)
A_sum_axis0 = A.sum(axis=0)  

# 按轴1（行方向）求和，输出形状丢失轴1维度 → (5,)
A_sum_axis1 = A.sum(axis=1)  

# 对轴0和轴1同时求和 → 标量（所有元素总和）
A.sum(axis=[0, 1])  

# 保持维度：求和后维度变为1，形状为 (5, 1)
sum_A = A.sum(axis=1, keepdim=True)
```

### 2. 均值计算

```Python

# 所有元素的均值
A.mean()  

# 按轴0（列方向）求均值 → (4,)
A.mean(axis=0)
```

---

## 四、累积和（cumsum）

沿指定轴计算**累积总和**，每一行/列是前面所有行/列的累加结果：

```Python

# 按轴0（行方向）计算累积和
B = A.cumsum(axis=0)
# 结果示例：
# tensor([[ 0,  1,  2,  3],
#         [ 4,  6,  8, 10],
#         [12, 15, 18, 21],
#         [24, 28, 32, 36],
#         [40, 45, 50, 55]])
```

---

## 五、点积与矩阵乘法

### 1. 向量点积

```Python

y = torch.ones(4, dtype=torch.float32)  # [1,1,1,1]

# 方法1：直接调用 dot
torch.dot(x, y)  

# 方法2：等价于按元素乘后求和
torch.sum(x * y)
```

### 2. 矩阵-向量乘法（mv）

```Python

# A 形状 (5,4)，x 形状 (4,) → 结果形状 (5,)
torch.mv(A, x)
```

### 3. 矩阵-矩阵乘法（mm）

```Python

# A 形状 (5,4)，B 形状 (4,5) → 结果形状 (5,5)
torch.mm(A, A.T)
```

---

## 六、范数（Norm）

范数用于衡量向量/矩阵的“大小”，PyTorch 中常用以下几种：

### 1. 向量 L2 范数

 $\Vert u \Vert_2 = \sqrt{u_1^2 + u_2^2 + ... + u_n^2}$ 

```Python

u = torch.tensor([3.0, -4.0])
torch.norm(u)  # 结果：5.0（√(3²+(-4)²)）
```

### 2. 向量 L1 范数

 $\Vert u \Vert_1 = |u_1| + |u_2| + ... + |u_n|$ 

```Python

torch.abs(u).sum()  # 结果：7.0（|3| + |-4|）
```

### 3. 矩阵 Frobenius 范数

 $\Vert A \Vert_F = \sqrt{\sum_{i,j} A_{i,j}^2}$ 

```Python

# 4行9列全1矩阵，元素总数 36 → 范数为 √36 = 6.0
torch.norm(torch.ones((4, 9)))
```

---

## 核心总结

|操作|PyTorch 代码示例|说明|
|---|---|---|
|矩阵转置|`A.T`|行与列互换|
|Hadamard 积|`A * B`|对应位置元素相乘|
|按轴求和|`A.sum(axis=0, keepdim=True)`|`keepdim=True` 保持维度不变|
|累积和|`A.cumsum(axis=0)`|沿轴方向累加|
|向量点积|`torch.dot(x, y)` / `torch.sum(x*y)`|一维向量专用|
|矩阵乘法|`torch.mm(A, B)` / `torch.mv(A, x)`|矩阵-矩阵/矩阵-向量乘法|
|L2 范数|`torch.norm(u)`|向量欧氏距离|
|Frobenius 范数|`torch.norm(A)`|矩阵元素平方和的平方根|

> （注：文档部分内容可能由 AI 生成）
