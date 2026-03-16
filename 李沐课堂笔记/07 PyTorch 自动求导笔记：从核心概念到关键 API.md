# PyTorch 自动求导笔记：从核心概念到关键 API

# PyTorch 自动求导笔记

## 一、核心概念

- **自动求导**：基于**向量链式法则**，计算函数在指定点的导数值，区别于符号求导（得到解析式）和数值求导（近似计算）。

- **计算图**：将计算拆解为操作子，表达为无环图（树状结构），分为：

    - **正向构造**：从输入 `x` 到输出 `y`，记录所有操作，时间复杂度  $O(n)$ ，内存复杂度  $O(1)$ 。

    - **反向传播**：从输出 `y` 到输入 `x`，计算梯度，时间复杂度  $O(n)$ ，内存复杂度  $O(n)$ （需保存中间结果）。

- **隐式构造**：PyTorch 会自动根据 Python 控制流（条件、循环、函数调用）动态构建计算图。

---

## 二、基础实现（实现1）

```Python

import torch

# 1. 创建张量并开启梯度追踪
x = torch.arange(4.0)
x.requires_grad_(True)  # 等价于 x = torch.arange(4.0, requires_grad=True)
x.grad  # 初始为 None，用于存储梯度

# 2. 定义计算图：y = 2 * (x₁² + x₂² + x₃² + x₄²)
y = 2 * torch.dot(x, x)

# 3. 反向传播求导
y.backward()

# 4. 验证梯度：∂y/∂x = 4x
print(x.grad)  # tensor([0., 4., 8., 12.])
print(x.grad == 4 * x)  # tensor([True, True, True, True])
```

---

## 三、梯度累积与清零（实现2）

PyTorch 默认会**累积梯度**，因此每次反向传播前需手动清零。

### 示例1：求和的梯度

```Python

x.grad.zero_()  # 清零之前的梯度
y = x.sum()
y.backward()
print(x.grad)  # tensor([1., 1., 1., 1.])，sum 的导数为 1
```

### 示例2：非标量的梯度

对非标量 `y` 调用 `backward()` 时，需传入 `gradient` 参数（指定微分函数的梯度），通常传入全 1 张量等价于对 `y` 求和后求导。

```Python

x.grad.zero_()
y = x * x
# 等价写法1：传入 gradient 参数
y.backward(torch.ones(len(x)))
# 等价写法2：先求和再反向传播
# y.sum().backward()

print(x.grad)  # tensor([0., 2., 4., 6.])，即 ∂(x²)/∂x = 2x
```

---

## 四、脱离计算图：`detach()`

使用 `.detach()` 将张量从计算图中分离，使其视为**常数**，不再参与梯度计算。

```Python

x.grad.zero_()
y = x * x
u = y.detach()  # u 脱离计算图，成为常数
z = u * x

z.sum().backward()
print(x.grad == u)  # tensor([True, True, True, True])，∂z/∂x = u

# 验证 y 的梯度计算不受影响
x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)  # tensor([True, True, True, True])
```

---

## 五、Python 控制流的梯度计算

PyTorch 支持**动态计算图**，即使函数包含 `if`/`while` 等控制流，仍能自动求导。

```Python

def f(a):
    b = a * 2
    while b.norm() < 1000:  # 循环控制流
        b = b * 2
    if b.sum() > 0:  # 条件控制流
        c = b
    else:
        c = 100 * b
    return c

# 初始化并求导
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

# 验证梯度：∂d/∂a = d / a
print(a.grad == d / a)  # tensor(True)
```

---

## 六、关键 API 总结

|API|作用|
|---|---|
|`requires_grad=True`|开启张量的梯度追踪|
|`.backward()`|反向传播计算梯度，非标量需传 `gradient`|
|`.grad.zero_()`|清零梯度，避免累积|
|`.detach()`|分离张量，脱离计算图，作为常数使用|
|`.grad`|存储张量的梯度值（初始为 `None`）|
> （注：文档部分内容可能由 AI 生成）
