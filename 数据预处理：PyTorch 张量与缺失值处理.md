# 数据预处理：PyTorch 张量与缺失值处理

# 04 数据操作 + 数据预处理（笔记整理）

---

## 一、PyTorch 张量（Tensor）基础操作

### 1. 张量核心概念

- 张量是 **N维数组**，是机器学习/神经网络的核心数据结构，由 `torch` 库实现。

- 本质：由数值组成的多维数组，可存储梯度等计算信息。

### 2. 张量创建

|方法|说明|
|---|---|
|`torch.arange()`|生成序列张量（类似 Python `range`）|
|`torch.zeros(shape)`|生成指定形状的全0张量（如 `torch.zeros((2,3,4))`）|
|`torch.ones(shape)`|生成指定形状的全1张量|
|`torch.tensor(列表)`|将 Python 列表直接转换为张量|
### 3. 张量属性与形状操作

- **形状**：`x.shape` → 查看张量维度信息

- **元素总数**：`x.numel()` → 统计张量中所有元素的个数

- **形状重塑**：`x.reshape(...)` → 不改变元素数量和总量，仅调整维度（如 `x.reshape(,)` 自动推导维度）

### 4. 索引与切片

- 基础切片：`[1:3, ]` → 行取索引1~2（左闭右开，不包含3）

- 跳跃访问：`[::3, ::2]` → 每3行取1行、每2列取1列

- 特殊索引：

    - `X[-1]` → 取最后一行

    - `X[1:3]` → 取第1~2行

### 5. 张量运算

#### 按元素运算

```Python

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # ** 为幂运算
```

- 支持 `+`、`-`、`*`、`/`、`**` 等逐元素操作。

#### 拼接与广播

- **拼接**：`torch.cat(张量列表, dim=0/1)`

    - `dim=0` → 按行拼接（增加行数）

    - `dim=1` → 按列拼接（增加列数）

- **广播机制**：不同维度的张量会自动扩展至兼容形状后再运算。

#### 内存优化

- 默认赋值（`Y = X + Y`）会开辟**新内存**。

- 原地执行（节省内存）：

    - `Y[:] = X + Y` → 切片赋值，复用原内存

    - `X += Y` → 直接在原张量上修改（前提：后续不再复用原 `X`）

### 6. 张量与 NumPy 互转

- 张量 → NumPy：`A = X.numpy()`

- NumPy → 张量：`B = torch.tensor(A)`

- 类型验证：

    ```Python
    
    type(A), type(B)  # 输出 (numpy.ndarray, torch.Tensor)
    ```

### 7. 标量转换

将**大小为1的张量**转为 Python 标量：

```Python

a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
# 输出：(tensor([3.5000]), 3.5, 3.5, 3)
```

- `a.item()`：直接提取标量值

- `float(a)` / `int(a)`：强制类型转换

---

## 二、数据预处理（Pandas + 张量转换）

### 1. 数据集创建与加载

#### 创建 CSV 数据集

```Python

import os
# 创建data文件夹，存放数据集文件
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
# 拼接文件路径：上级目录/data/house_tiny.csv
data_file = os.path.join('..', 'data', 'house_tiny.csv')
# 写入CSV文件，定义列名+4行样本数据
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 表头：房间数、巷子类型、房价
    f.write('NA,Pave,127500\n')        # 样本1：房间数缺失
    f.write('2,NA,106000\n')          # 样本2：巷子类型缺失
    f.write('4,NA,178100\n')          # 样本3
    f.write('NA,NA,140000\n')         # 样本4：全部缺失
```

#### 加载数据集

```Python

import pandas as pd
# 读取CSV文件，生成DataFrame表格数据（pandas核心数据结构）
data = pd.read_csv(data_file)
```

### 2. 缺失值处理

```Python

# 拆分数据：前2列为输入特征（房间数、巷子），最后1列为输出标签（房价）
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]

# 1. 处理数值型缺失值：NumRooms是数字特征，用【列均值】填充NA
# 均值填充是最常用的数值缺失处理方法，不破坏数据分布
inputs = inputs.fillna(inputs.mean())

# 2. 处理类别型缺失值：Alley是文字特征，无法填均值
# pd.get_dummies：将文字/缺失值转为0/1数字（独热编码）
# dummy_na=True：把缺失值NA单独作为一个类别编码
inputs = pd.get_dummies(inputs, dummy_na=True)
```

### 3. 转换为张量格式

```Python

import torch
# 先将pandas表格转为numpy数组，再转为PyTorch张量
# 神经网络模型仅支持张量格式输入
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
```

---

## 三、核心总结

1. **张量是 PyTorch 的核心数据结构**，支持多维数组运算、自动微分与内存优化。

2. **数据预处理流程**：创建/加载数据集 → 处理缺失值（插值+独热编码）→ 转换为张量。

3. **内存优化技巧**：优先使用原地操作（`X += Y` 或切片赋值）减少内存开销。

---
> （注：文档部分内容可能由 AI 生成）