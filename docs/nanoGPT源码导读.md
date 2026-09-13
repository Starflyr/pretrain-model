# nanoGPT 源码导读：从一个字符走到一次参数更新

这份文档配合本项目里的 `nanoGPT/` 阅读，目标是让你能亲自跟着源码解释：文本如何变成输入，Transformer 如何预测下一个字符，训练脚本如何更新参数，以及训练后的模型如何生成文本。

读者假设：你有编程经验，能读 Python 的函数、类、循环，但不要求已经熟悉深度学习。公式都对应具体变量和张量形状。本文按理解顺序拆解代码，不按文件从第一行机械复述到最后一行。

**源码基线**：本地记录的 nanoGPT 提交为 `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`。本文写于 2026-09-12，行号针对当前本地文件，后续更新源码时可能变化。文中的运行配置来自项目已有的 [预训练指南](../learning-guide-4gb/01-环境与预训练.md)。本文分析本地实现，不代表上游最新版本，也不代表已经完成模型训练。

## 阅读导航

1. [先看全貌与阅读路线](#1-先看全貌与阅读路线)
2. [明确本次模型配置](#2-明确本次模型配置)
3. [文本如何变成数字](#3-文本如何变成数字)
4. [训练样本为什么错开一位](#4-训练样本为什么错开一位)
5. [配置如何进入训练程序](#5-配置如何进入训练程序)
6. [GPT 的整体结构](#6-gpt-的整体结构)
7. [逐行理解一个 Transformer Block](#7-逐行理解一个-transformer-block)
8. [多头因果注意力详解](#8-多头因果注意力详解)
9. [输出与交叉熵损失](#9-输出与交叉熵损失)
10. [参数初始化与参数量](#10-参数初始化与参数量)
11. [训练启动与三种初始化路径](#11-训练启动与三种初始化路径)
12. [一次训练迭代到底做什么](#12-一次训练迭代到底做什么)
13. [验证、日志与 checkpoint](#13-验证日志与-checkpoint)
14. [生成文本的完整路径](#14-生成文本的完整路径)
15. [多卡、编译与性能代码](#15-多卡编译与性能代码)
16. [把源码对应到你的机器](#16-把源码对应到你的机器)
17. [自测题与建议阅读顺序](#17-自测题与建议阅读顺序)
18. [关键位置索引与核验说明](#18-关键位置索引与核验说明)

## 1. 先看全貌与阅读路线

### 1.1 核心其实只有几份文件

| 文件 | 负责什么 | 阅读时抓住的问题 |
|---|---|---|
| [data/shakespeare_char/prepare.py](../nanoGPT/data/shakespeare_char/prepare.py) | 下载文本、构造字符表、切分并编码 | 字符怎么变成 token ID？ |
| [configurator.py](../nanoGPT/configurator.py) | 用配置文件和命令行覆盖变量 | 最终参数究竟从哪里来？ |
| [model.py](../nanoGPT/model.py) | 定义网络、损失、优化器分组与生成方法 | 给定一批 ID，如何得到预测？ |
| [train.py](../nanoGPT/train.py) | 取数据、建模型、训练、验证、保存 | 谁在什么时间调用模型？ |
| [sample.py](../nanoGPT/sample.py) | 加载模型与字符表、编码提示、输出生成结果 | checkpoint 如何变成文字？ |
| [config/train_shakespeare_char.py](../nanoGPT/config/train_shakespeare_char.py) | Shakespeare 实验的参数覆盖 | 示例配置和本机配置有何差别？ |

`model.py` 负责计算；`train.py` 负责安排计算何时发生。模型本身不知道训练文件在哪，也不负责决定要训练多少步。

### 1.2 数据流总览

```text
input.txt：原始文本
    │ prepare.py：字符映射 + 切分 + 编码
    ├── train.bin：训练 token 流
    ├── val.bin：验证 token 流
    └── meta.pkl：字符与 ID 的对应关系
                 │
train.py：get_batch() 随机截取窗口
    │
    ├── X：[B,T]，输入 token ID
    └── Y：[B,T]，向后错开一位的目标 ID
                 │
model.py：GPT.forward(X,Y)
    │  token embedding + position embedding
    │  多层 Transformer Block
    │  最终 LayerNorm + 输出投影
    ├── logits：[B,T,V]
    └── loss：标量
                 │
train.py：backward → 梯度裁剪 → optimizer step
                 │
              ckpt.pt
                 │
sample.py：提示文字 → ID → GPT.generate() → ID → 文字
```

这里的“预训练”是从随机权重出发，用文本本身提供监督信号。目标不是人为标注的“正确答案”，而是文本中实际出现的下一个字符。

### 1.3 先建立三个对象的区别

| 对象 | 可以怎样理解 | 是否参与参数更新 |
|---|---|---|
| 数据 `X, Y` | 本次练习题与标准答案 | 数据本身不更新 |
| 参数 `nn.Parameter` | 模型学习的数值，例如 Linear 的权重 | 优化器更新它们 |
| 激活 `x, q, k, logits` | 当前这次计算产生的中间结果 | 用于计算梯度，不是持久模型权重 |

同一个变量名 `x` 在不同函数里含义不同：`get_batch()` 中是整数 ID；注意力层中已经是浮点特征。看源码时，形状与 dtype 比变量名字更可靠。

## 2. 明确本次模型配置

后文统一使用已有指南的单卡小模型配置举例：

| 符号 | 源码参数 | 本文数值 | 含义 |
|---|---|---:|---|
| B | `batch_size` | 4 | 一次前向同时处理 4 段文本 |
| T | `block_size` | 128 | 训练窗口长度为 128 个字符 |
| C | `n_embd` | 128 | 每个位置用 128 个浮点特征表示 |
| H | `n_head` | 4 | 注意力头数 |
| D | `C // H` | 32 | 每个头的特征维度 |
| L | `n_layer` | 4 | 堆叠 4 个 Block |
| V | `vocab_size` | 65 | 当前字符表大小 |
| A | `gradient_accumulation_steps` | 4 | 累积 4 个微批次再尝试更新参数 |

其他有效配置：`bias=False`、`dropout=0.2`、`dtype='float16'`、`compile=False`。其中 `dropout=0.2` 来自 Shakespeare 配置，`bias=False` 来自 `train.py` 默认值。

**不要把三套默认值混为一谈。** `GPTConfig` 类自己的默认值、`train.py` 的默认值、Shakespeare 配置文件的值并不完全相同。实际训练由最后传入构造函数的参数决定。

## 3. 文本如何变成数字

阅读 [prepare.py](../nanoGPT/data/shakespeare_char/prepare.py)，重点是第 24—61 行。

### 3.1 字符词表

```python
chars = sorted(list(set(data)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
```

`set` 去重，`sorted` 给出稳定的排列，随后按顺序分配整数 ID。空格、换行、大写字母、小写字母、标点都是独立字符。

假设一个玩具文本只包含 `a、b、c`，编码可以是 `a→0、b→1、c→2`。这些编号没有大小语义：ID 2 不代表比 ID 1 更重要，也不表示字符之间的相似程度。相似关系由后面的可训练 embedding 表示。

`encode()` 做字符查表，`decode()` 做反向查表。这两个函数没有神经网络，也没有训练过程。

### 3.2 切分与落盘

```python
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]
```

它按原始文本顺序取前 90% 为训练集、后 10% 为验证集，没有先随机打乱字符。打乱字符会破坏语言的连续结构。

编码之后用 `np.uint16` 保存到 `.bin`：每个 token 占 2 字节。这个格式没有字段名、形状信息或 tokenizer 定义；读取方必须自己约定 dtype。

当前项目的文件大小对应：

| 文件 | 字节数 | token 数 |
|---|---:|---:|
| `train.bin` | 2,007,708 | 1,003,854 |
| `val.bin` | 223,080 | 111,540 |

`meta.pkl` 保存 `vocab_size`、`stoi` 和 `itos`。它和模型共同构成可解释的结果：只有权重而没有正确字符映射，生成的 ID 就无法可靠还原成文字。

### 3.3 这份准备脚本的边界

词表在切分之前从整份文本提取，所以验证集的字符种类也进入词表。这是教学简化；严谨实验可以只从训练文本拟合词表，再明确未知字符如何处理。这里没有 `<unk>` 回退，编码未收录字符会查表失败。

代码注释写了 “train and test splits”，但变量与文件实际是 `train` 和 `val`，没有第三份独立测试集。

`uint16` 只能容纳 0—65535。更换为大词表 tokenizer 时，必须检查最大 ID；写入和读取的 dtype 要一起调整。

### 3.4 其他数据目录做什么

`data/shakespeare/prepare.py` 使用相同题材的文本，但用 GPT-2 BPE 编码，不是字符编码。`data/openwebtext/prepare.py` 则对文档做 BPE 编码，并在每篇后追加结束 token，再拼接写入大型文件。

训练器看到的都是整数流，但整数对应的语言单位不同。不能把字符模型的 checkpoint 与 BPE tokenizer 混用，也不能直接比较两种 token 单位下的 loss 数字。

## 4. 训练样本为什么错开一位

阅读 `train.py` 第 116 行的 `get_batch()`。

### 4.1 内存映射与随机窗口

```python
data = np.memmap(path, dtype=np.uint16, mode='r')
ix = torch.randint(len(data) - block_size, (batch_size,))
```

`memmap` 让程序像操作数组一样访问磁盘数据，不必先把整个 token 文件复制成一个普通内存数组。操作系统仍会使用内存缓存，后续切片和 dtype 转换也会分配内存，因此它不等于“零内存占用”。

每次随机选择 B 个起点，允许重复和窗口重叠。这里没有 DataLoader、没有遍历完所有样本的 epoch 循环，也没有保证每个位置恰好出现一次。

为什么起点上界是 `len(data) - block_size`？因为除 X 的 T 个 token 外，Y 还要读到后面第 T+1 个 token；`torch.randint` 的上界不包含在取值范围里。

### 4.2 X 与 Y 的关系

```python
x = data[i:i + block_size]
y = data[i + 1:i + 1 + block_size]
```

假设抽取到的原文为 `hello`，窗口 T=4：

```text
位置       0    1    2    3
X          h    e    l    l
Y          e    l    l    o

第 0 位：看到 h，预测 e
第 1 位：看到 he，预测 l
第 2 位：看到 hel，预测 l
第 3 位：看到 hell，预测 o
```

这一次前向同时提供 4 个预测任务。后文的因果 mask 保证第 0 位不能偷看输入第 1 位已经存在的 `e`。

**标签右移已经由取数据完成。** `GPT.forward()` 不会再帮你做一次 shift，它直接对齐 `logits[b,t]` 与 `targets[b,t]`。

### 4.3 为什么转成 int64

磁盘保存时用 `uint16` 节省空间，送进模型前转成 `np.int64`，得到 PyTorch 的长整型索引张量。ID 用来查 embedding，不会因为训练选择 FP16 就变成半精度浮点数。

`torch.stack()` 把 B 个 `[T]` 窗口叠成 `[B,T]`。本机例子中 X 和 Y 都是 `[4,128]`。

CUDA 分支使用 `pin_memory().to(device, non_blocking=True)`，以固定页内存支持异步传输。这是传输路径的优化；实际与 GPU 计算重叠多少，要看运行时调度与硬件。

## 5. 配置如何进入训练程序

阅读 `train.py` 第 34—78 行与整个 `configurator.py`。

### 5.1 它不是 argparse

训练文件先定义全局变量，然后执行：

```python
exec(open('configurator.py').read())
```

配置器在同一个全局命名空间中运行。它按命令行顺序处理参数：不含 `=` 的项被视为 Python 配置文件并执行；含 `=` 的项被视为 `--key=value`，通过 `literal_eval` 尝试解析布尔、整数、浮点等。

例如：

```text
train.py 默认 batch_size=12
    ↓ 执行 config/train_shakespeare_char.py
batch_size=64
    ↓ 处理 --batch_size=4
batch_size=4
```

**准确规则是后执行的覆盖先执行的。** 常见写法是“配置文件在前、命令行覆盖在后”；如果把配置文件放在最后，它也能覆盖前面的参数。

### 5.2 容易忽略的解析规则

- 命令格式是 `--batch_size=4`，不能照搬成 `--batch_size 4`。
- 有严格的类型相等检查。原值是浮点时，传 `--dropout=0` 会与整数类型冲突，应写 `--dropout=0.0`。
- `--compile=False` 用 Python 布尔写法；小写 `false` 会成为字符串并导致类型检查失败。
- 未知变量名会抛异常。
- 实现使用 `arg.split('=')`，参数值本身再含 `=` 时可能解包失败。
- 配置文件是会执行的 Python 代码，应使用自己理解和信任的配置。

`config_keys` 在执行配置前收集基本类型的变量名，之后生成日志用的 `config` 字典。它是当时配置的快照，后续恢复 checkpoint、DDP 调整等动作不一定同步修改这份快照。

### 5.3 当前工作目录也属于运行前提

`train.py` 用相对路径寻找 `configurator.py`、`data/` 和输出目录。因此按指南运行时，工作目录应在 `nanoGPT/`。仅从别处用绝对路径调用 `train.py`，不会自动把工作目录切过去。

## 6. GPT 的整体结构

阅读 `model.py` 第 109—193 行。先看 `GPT.__init__()` 与 `GPT.forward()`，再回头读前面的层定义，会更容易理解。

### 6.1 构造函数建立可训练组件

```python
wte = nn.Embedding(config.vocab_size, config.n_embd)
wpe = nn.Embedding(config.block_size, config.n_embd)
h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
ln_f = LayerNorm(config.n_embd, bias=config.bias)
```

`ModuleDict`、`ModuleList` 不只是普通容器：放进去的子模块会被 PyTorch 正确注册，随后 `.parameters()`、`.to(device)`、`.state_dict()` 能遍历它们。

`wte` 是 token embedding，权重形状 `[V,C]=[65,128]`。输入 ID 选择对应的一行向量。`wpe` 是可学习的绝对位置 embedding，形状 `[128,128]`；这里没有 RoPE。

### 6.2 输入进入 forward

```python
b, t = idx.size()
pos = torch.arange(0, t, dtype=torch.long, device=device)
tok_emb = self.transformer.wte(idx)
pos_emb = self.transformer.wpe(pos)
x = self.transformer.drop(tok_emb + pos_emb)
```

| 变量 | 形状 | 意义 |
|---|---|---|
| `idx` | `[4,128]` | 字符 ID |
| `pos` | `[128]` | 当前窗口内的 0—127 位置 |
| `tok_emb` | `[4,128,128]` | 字符的可学习表示 |
| `pos_emb` | `[128,128]` | 每个位置的可学习表示 |
| 相加后的 `x` | `[4,128,128]` | 同时包含字符与位置信息 |

相加通过广播让每条样本使用相同的位置表。位置从当前窗口的 0 开始，不是原始文件中的全局字符偏移。

同一个字符在不同位置，token embedding 相同，但 position embedding 不同。随着注意力层加入上下文，即使同一个字符出现在相同相对位置，最终特征也可能不同。

### 6.3 主干保持形状

```python
for block in self.transformer.h:
    x = block(x)
x = self.transformer.ln_f(x)
```

4 个 Block 的输入输出都是 `[4,128,128]`。形状不变不代表内容不变：每一层都重新处理上下文信息，构造更适合预测的特征。

整个网络属于 decoder-only、因果自注意力结构，没有额外 encoder，也没有对另一组编码结果做 cross-attention。

## 7. 逐行理解一个 Transformer Block

阅读 `model.py` 第 18—27 行与第 78—106 行。

### 7.1 两行代码就是一个 Block

```python
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
```

展开为：

```text
输入 x
  ├──────────────────┐
  └→ LayerNorm → Attention → 相加 → x1
                                  ├─────────────────┐
                                  └→ LayerNorm → MLP → 相加 → 输出
```

先归一化，再进注意力或 MLP，这种顺序称为 Pre-LN。第二行使用的是第一行更新后的 `x`。

`x + ...` 是残差连接。子层学习在已有表示上加什么改变量；它也为梯度提供较直接的传播路径。相加两边必须形状一致，所以注意力与 MLP 最终都回到 C 维。

### 7.2 LayerNorm 沿哪个维度计算

本实现调用：

```python
F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
```

对于 `[B,T,C]` 的输入，归一化在最后 C 个特征上独立进行。对每个样本的每个位置分别计算均值和方差，不是把整批样本混在一起，也不沿时间维归一化。

可以写成：`normalized = (x - mean) / sqrt(variance + epsilon)`，再乘以可训练的 `weight`，有 bias 时再加可训练偏移。这里 `bias=False` 仅去掉偏移，缩放参数仍存在。

源码关于 PyTorch 是否支持可选 bias 的注释带有历史背景；阅读当前项目时关注这份自定义类的实际行为即可。

### 7.3 MLP 为什么先放大再缩小

```text
[B,T,128]
  → Linear(128,512)
  → GELU
  → Linear(512,128)
  → Dropout
  → [B,T,128]
```

MLP 在每个位置上使用同一套参数独立处理特征，不直接混合不同时间位置。跨位置的信息交流主要发生在注意力层。

放大到 `4*C` 为非线性变换提供更宽的中间表示。GELU 引入非线性，否则两个线性映射可以合并成一个线性映射。Dropout 在训练时随机屏蔽并缩放部分输出，在 `eval()` 模式下关闭。

注意“对每个位置独立”不等于“不包含上下文”：MLP 的输入已经经过注意力混合，带有可见前缀的信息。

## 8. 多头因果注意力详解

阅读 `model.py` 第 29—76 行。这是最值得手工跟形状的部分。

### 8.1 一次线性映射获得 Q、K、V

```python
self.c_attn = nn.Linear(C, 3 * C, bias=config.bias)
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
```

输入 `[4,128,128]` 先变成 `[4,128,384]`，再切成三个 `[4,128,128]`。

- Q（query）：当前位置用来询问其他位置的表示。
- K（key）：每个位置用于与查询匹配的表示。
- V（value）：匹配后实际汇总的内容表示。

它们都来自同一个输入 x，但经过不同的可训练权重投影。可以把这三种用途作为理解入口，但不要把某个维度固定解释成“语法”或“人物”等人工标签。

PyTorch `Linear(C,3C)` 的权重存储形状是 `[3C,C]`，前向相当于 `x @ weight.T`，不是直接按存储顺序做 `x @ weight`。

### 8.2 拆成多个头

```python
q = q.view(B, T, H, C // H).transpose(1, 2)
```

```text
[4,128,128]
  → view：[4,128,4,32]
  → transpose：[4,4,128,32]
                    ↑   ↑
                 时间   每头维度
```

K、V 做相同变换。`assert C % H == 0` 保证能整齐切开。

多头不是把输入文本分成四段。每个头都覆盖所有 T 个位置，但使用不同的特征子空间计算关联。

### 8.3 QK 转成相关性分数

手写分支为：

```python
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
```

形状：`[B,H,T,D] @ [B,H,D,T] → [B,H,T,T]`。

其中 `att[b,h,i,j]` 表示第 b 条样本、第 h 个头中，位置 i 对位置 j 的匹配分数。除以 `sqrt(D)` 是为了控制点积的尺度，避免维度增大后 softmax 过于尖锐。

### 8.4 因果 mask 阻止偷看未来

```text
          被关注的位置 j
           0  1  2  3
查询 i=0   ✓  ×  ×  ×
查询 i=1   ✓  ✓  ×  ×
查询 i=2   ✓  ✓  ✓  ×
查询 i=3   ✓  ✓  ✓  ✓
```

允许看自己及之前的位置。因为位置 i 的输入是当前字符，目标是第 i+1 个字符，所以看自己是正确的，看下一位才会泄漏答案。

手写实现用下三角矩阵，把未来位置填成 `-inf`，softmax 后对应概率就是 0。这里注册的 `bias` 是 mask buffer，和 Linear 中的可训练 bias 不是一回事。

buffer 会跟随模型移动设备并可进入状态字典，但不会作为参数交给优化器学习。

### 8.5 softmax 后汇总 V

```python
att = F.softmax(att, dim=-1)
att = self.attn_dropout(att)
y = att @ v
```

softmax 沿“被关注的位置 j”计算，让每个查询位置获得一组权重。未施加 dropout 时，每行权重和为 1；训练态经过 dropout 后，不保证每行仍严格求和为 1。

若某行权重是 `[0.2,0.3,0.5,0]`，该位置的输出就是 `0.2*v0 + 0.3*v1 + 0.5*v2`。

`[B,H,T,T] @ [B,H,T,D] → [B,H,T,D]`，本例回到 `[4,4,128,32]`。

### 8.6 合并注意力头

```python
y = y.transpose(1, 2).contiguous().view(B, T, C)
y = self.resid_dropout(self.c_proj(y))
```

先还原成 `[B,T,H,D]`，再拼接最后两个维度。`transpose()` 改变张量视图的步长布局，`contiguous()` 确保后面的 `view()` 可以按连续布局重解释。

拼接后，`c_proj: C→C` 学习怎样组合不同头的结果，最终返回 `[B,T,C]`。

### 8.7 本机主要走哪条分支

构造函数用 `hasattr(F, 'scaled_dot_product_attention')` 设置 `self.flash`。当前 PyTorch 具备该接口，因此会调用：

```python
F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    dropout_p=self.dropout if self.training else 0,
    is_causal=True,
)
```

`is_causal=True` 表达相同的因果约束；缩放、softmax 和乘 V 由接口内部处理。

**变量名 `flash` 不足以证明实际使用了 FlashAttention CUDA 内核。** 这段代码只检测 API 是否存在；实际后端由 PyTorch 按设备、dtype、形状等条件选择。理解数学时看手写分支，确认性能时需要实际分析运行后端。

## 9. 输出与交叉熵损失

### 9.1 从特征变成每个字符的分数

最终 `x` 为 `[B,T,C]`，`lm_head` 将 C 维映射到 V 维：

```python
logits = self.lm_head(x)
```

得到 `[4,128,65]`。`logits[b,t,:]` 是位置 t 对下一个字符的 65 个候选分数，还不是概率。

每次前向共有 `4*128=512` 个预测位置。输出 65 维不是一次生成 65 个字符，而是为下一个字符提供 65 个候选分数。

### 9.2 为什么要展平

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    targets.view(-1),
    ignore_index=-1,
)
```

`logits` 变成 `[512,65]`，`targets` 变成 `[512]`。交叉熵可以把它看成 512 道分类题，然后取有效位置的平均值。

单个位置的损失为 `-log(p_correct)`。如果正确字符概率为 0.5，损失约为 0.693；如果为 0.01，损失约为 4.605。正确字符概率越高，损失越低。

`F.cross_entropy` 内部完成相应的 log-softmax 计算，不需要在它前面自己调用 softmax。

`ignore_index=-1` 表示标签为 -1 的位置不计入 loss；当前字符数据取样没有生成这类标签，正常情况下全部位置都参与计算。

### 9.3 均匀预测能作什么参照

若 65 个字符完全均匀，每个目标概率为 `1/65`，则 loss 为 `ln(65)≈4.174`。这是理论参照，不是承诺随机初始化模型实测恰好等于这个数。

`exp(loss)` 可以转换成对应 token 单位的困惑度。本项目是字符级，不能拿这个数字与另一种 BPE 词表的困惑度直接比较。

### 9.4 不传 targets 时，输出形状变了

```python
logits = self.lm_head(x[:, [-1], :])
loss = None
```

只投影最后一个位置，输出 `[B,1,V]`。用列表 `[-1]` 保留时间维；写成 `x[:,-1,:]` 则会减少一维。

这仅省掉前面位置的输出投影。Transformer 主干仍计算整个输入窗口，没有因此自动获得 KV cache。

另外，**有没有 targets** 决定计算哪些 logits 和 loss；**train/eval 模式**决定 dropout 等行为，两者互相独立。验证时既可以 `model.eval()`，也可以传 targets 计算 loss。

## 10. 参数初始化与参数量

### 10.1 权重共享

```python
self.transformer.wte.weight = self.lm_head.weight
```

输入 embedding 和输出分类器使用同一个参数对象，形状都是 `[V,C]`。不是复制两份初值，而是后续更新也共同作用于这一份权重。

输入时根据 ID 取某一行；输出时用上下文特征与各行权重计算分数。共享减少了参数量，并把输入、输出的 token 表示关联起来。

### 10.2 随机初始化怎么做

Linear 与 Embedding 权重按标准差 0.02 的正态分布初始化，Linear 的偏置存在时置零。LayerNorm 的缩放初始为 1，偏移存在时为 0。

名字以 `c_proj.weight` 结尾的参数再用 `0.02/sqrt(2*L)` 的标准差初始化。注意力和 MLP 的输出投影都匹配这个后缀。这使残差分支的初始输出尺度更保守。

随机初始化不是“模型已经知道英语”；它只是给可训练参数一个起点。

### 10.3 手工算出本机小模型参数量

在 `bias=False` 时，每个 Block 包含：

| 部分 | 参数数公式 |
|---|---:|
| QKV 投影 | `3*C*C` |
| 注意力输出投影 | `C*C` |
| MLP 两个投影 | `4*C*C + 4*C*C` |
| 两个 LayerNorm 缩放 | `2*C` |
| 合计 | `12*C² + 2*C` |

整个模型总参数数：

```text
token embedding（含共享输出权重） VC       =   8,320
position embedding             TC       =  16,384
4 个 Block                     L(12C²+2C)= 787,456
最终 LayerNorm                  C        =     128
总计                                     = 812,288
```

`get_num_params()` 默认会减去 position embedding，但保留共享 token embedding，因此默认打印的计数是 **795,904，约 0.80M**。完整参数量约 0.81M，两者差别来自统计口径。

参数量不是总显存：还需要梯度、AdamW 状态、中间激活与运行时工作空间。`dtype=float16` 使用 autocast，也不意味着所有参数与优化器状态都以 FP16 存储。

## 11. 训练启动与三种初始化路径

阅读 `train.py` 第 82—212 行。

### 11.1 启动准备

脚本判断是否存在分布式 `RANK`，设置设备和主进程身份，创建输出目录，设置随机种子 `1337 + seed_offset`，然后创建 autocast 上下文。

模型初始化和随机窗口选择都会消耗随机数。验证也随机取窗口，所以改变验证次数可能改变后续训练抽到的样本。固定 seed 不等于任意改动配置后仍会逐步相同。

### 11.2 scratch

先读取 `meta.pkl` 的 `vocab_size`，组装 `GPTConfig`，再构造 GPT。这里是本次实验的路径。

如果找不到 meta，代码退回 50304 的词表大小，并不会自动推断当前 `.bin` 是字符编码。字符实验丢失 meta 时，即便某些计算还能执行，配置与解码也不再正确。

### 11.3 resume

加载 `out_dir/ckpt.pt`，从 checkpoint 强制恢复层数、头数、隐藏维度、上下文长度、bias 和词表大小，构造模型并加载参数，再恢复迭代计数、验证损失与 optimizer 状态。

代码会移除状态字典键名前缀 `_orig_mod.`，用于兼容编译包装带来的命名变化。

不是所有命令行参数都会被 checkpoint 覆盖，例如 dropout 可以使用当前指定值。学习率安排也由当前脚本的配置计算，并在循环里重新写入 optimizer 的参数组。

### 11.4 gpt2 系列

`GPT.from_pretrained()` 从 Transformers 加载 GPT-2 权重，限定几种已知架构。部分权重由于上游 Conv1D 与本实现 Linear 存储布局不同，需要转置再复制。

它强制使用 GPT-2 的词表大小、上下文长度和 bias 设置，只允许覆盖 dropout。这条路径不是当前字符模型从零预训练的路径，当前最小环境也没有为这条分支安装所有依赖。

### 11.5 crop_block_size

允许将位置 embedding 和手写 mask 截短，不能用它无条件扩长上下文。它改变的是模型可使用的位置范围，不会重新学习被截短后的模型。

普通 scratch 本来就使用目标 block size，通常不会进入这条裁剪分支。

## 12. 一次训练迭代到底做什么

阅读 `train.py` 第 231—333 行。这是把“模型定义”变成“学习过程”的关键。

### 12.1 先把完整循环压缩成伪代码

```python
X, Y = get_batch('train')
while True:
    设置当前学习率
    按间隔验证，满足条件则保存
    for micro_step in range(A):
        logits, loss = model(X, Y)
        loss = loss / A
        X, Y = get_batch('train')
        对缩放后的 loss 反向传播，累积梯度
    还原梯度尺度并裁剪
    尝试执行 AdamW 更新
    更新 GradScaler
    清空梯度
    记录日志，计数加一，判断结束
```

这里只是对源码的阅读展开，不是新增的可执行训练脚本。

### 12.2 学习率预热与余弦衰减

`get_lr(it)` 分三段：

1. `it < warmup_iters`：`learning_rate * (it+1)/(warmup_iters+1)`，逐渐增大。
2. 预热结束至 `lr_decay_iters`：按余弦从峰值下降到 `min_lr`。
3. 超过衰减结束：保持 `min_lr`。

`it=0` 的学习率是一个较小的正数，不是零。正式小模型配置中，峰值 `1e-3`、最低 `1e-4` 来自 Shakespeare 配置，指南把预热设为 100、衰减终点设为 2000。

学习率控制更新尺度，loss 衡量预测误差，两者不能相互替代。脚本每轮把计算出的 lr 写入全部 optimizer 参数组。

### 12.3 AdamW 如何选参数

`model.configure_optimizers()` 收集所有 `requires_grad=True` 参数，按维数分为两组：

- 二维及以上：启用 weight decay，包括线性权重和 embedding。
- 一维及以下：不启用 weight decay，包括 LayerNorm 缩放和存在的 bias。

AdamW 持有梯度的一阶、二阶移动统计，利用历史信息调整参数更新，并对指定组执行权重衰减。`beta1`、`beta2` 控制统计的历史保留程度。

若 AdamW 接口提供 `fused` 参数且设备为 CUDA，代码请求 fused 版本，以减少一些优化器操作的开销。这不改变优化器组的学习任务。

### 12.4 梯度累积为什么除以 A

本例一次微批次处理 `B*T=512` 个目标，4 个微批次合计 2048 个目标。每个微批次单独前向、反向，最后才更新一次。

设 4 次各自的平均损失为 `L1…L4`，希望累积的是平均损失的梯度：

```text
∇[(L1+L2+L3+L4)/4] = ∇L1/4 + ∇L2/4 + ∇L3/4 + ∇L4/4
```

所以每次先除以 4。`.backward()` 默认把结果累加到参数的 `.grad`，不会自动覆盖上一微批次的梯度。若每个微批次都清空梯度，就失去了累积效果。

这能降低同时保留的激活量；参数、参数梯度和优化器状态仍需常驻。由于 dropout、数值计算与随机样本等因素，不应承诺它与另一种执行方式逐位相同。

### 12.5 前向、反向和更新是不同动作

`model(X,Y)` 计算预测和 loss，同时在启用梯度时建立计算图；`loss.backward()` 沿图计算参数梯度；`optimizer.step()` 才使用梯度修改参数。

训练器在当前前向之后立即取下一批数据，再对当前 loss 做 backward。虽然变量 X、Y 已重新绑定，当前计算图仍保有反向所需的内容，不会把新标签误用于旧 loss。

### 12.6 autocast 与 GradScaler 分别解决什么

`with ctx:` 在 CUDA 上启用 autocast，让不同算子按规则选择运算精度。模型没有在这里整体 `.half()`，整数 ID 也仍是整数。

FP16 表示很小的梯度时可能下溢。GradScaler 在反向前放大 loss，让梯度随之放大，更新前再还原；遇到非有限梯度时可以跳过参数更新，并调整后续缩放尺度。

源码顺序为：

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad(set_to_none=True)
```

`unscale_()` 在启用裁剪时执行。必须先还原，再裁剪，否则阈值作用于人为放大的梯度，改变了裁剪含义。没有显式 unscale 时，scaler 的 step 路径负责相应处理。

### 12.7 梯度裁剪与清零

`clip_grad_norm_` 按整体梯度范数限制大小，必要时等比例缩小梯度，不是将每一个梯度元素直接截到 `[-1,1]`。

`zero_grad(set_to_none=True)` 将梯度引用清空，避免下一轮继续累积上一轮结果，并减少部分清零开销。它不把模型参数清零。

### 12.8 重要边界：max_iters 不是循环执行次数

源码在一轮更新之后执行：

```python
iter_num += 1
if iter_num > max_iters:
    break
```

从 `iter_num=0` 启动时，会执行编号 0 到 `max_iters` 的循环。因此：

| 设置 | 更新循环次数 | 不发生 AMP 跳步时的参数更新次数 |
|---|---:|---:|
| `max_iters=100` | 101 | 101 |
| `max_iters=2000` | 2001 | 2001 |

如果只谈指南的 2000 步预算，`2000*2048=4,096,000` token；按这份代码完整循环计算是 `2001*2048=4,098,048` token。两者都包含重复采样，也不包括额外验证计算。

## 13. 验证、日志与 checkpoint

### 13.1 estimate_loss 做了什么

`train.py` 第 216 行的函数先 `model.eval()`，然后分别从 train 和 val 随机取 `eval_iters` 批，计算平均 loss，最后 `model.train()`。

`@torch.no_grad()` 关闭梯度记录；`eval()` 关闭 dropout 等训练行为。只调用 `eval()` 不会自动关闭 autograd；只调用 `no_grad()` 也不会自动关闭 dropout。

验证仍然是抽样估计，不是遍历完整验证集。源码注释中 “arbitrarily accurate” 不应理解为有限次抽样保证精确。

`eval_iters=20` 时，一轮评估会做 20 次训练集前向和 20 次验证集前向，共 40 次，不能只按 20 次计算开销。

### 13.2 两种 loss 日志不是同一个统计量

```text
step 200: train loss ... val loss ...
iter 200: loss ... time ... mfu ...
```

第一行来自评估函数：eval 模式、多批平均。第二行来自训练循环：训练模式，最后一个微批次的 loss 乘回累积次数。

因此 `iter ... loss` **不是所有累积微批次的平均 loss**，也不能要求它与评估的 train loss 完全一致。评估区间内耗费的时间还可能进入该轮 `dt`，不能拿每一条 time 直接代表纯训练内核耗时。

### 13.3 checkpoint 在更新之前保存

每轮顺序是“设学习率 → 验证/保存 → 更新参数”。所以 `step 100` 的检查点代表已经完成编号 0—99 的 100 轮更新之后、编号 100 更新之前的状态。

```text
iter_num=0    评估初始模型，不保存 → 更新 #1
iter_num=1                         → 更新 #2
...
iter_num=100  评估，条件满足则保存 → 更新 #101 → 退出
```

代码没有退出前无条件保存。最后一次内存里的更新可能没有进入磁盘 checkpoint。

### 13.4 保存条件与 best_val_loss

```python
if losses['val'] < best_val_loss or always_save_checkpoint:
    best_val_loss = losses['val']
    if iter_num > 0:
        # 保存 ckpt.pt
```

Shakespeare 配置设 `always_save_checkpoint=False`，只有验证 loss 改善才保存，而且 step 0 不保存。若之后一直没有优于初始评估，输出目录可能存在却没有 checkpoint。

设为 True 时，每次评估都会更新 `best_val_loss` 为当前值，它就不一定是历史最小值。文件始终写到同一个 `ckpt.pt`，没有自动保留多个历史版本。

### 13.5 保存了什么，没保存什么

| 字段 | 用途 |
|---|---|
| `model` | 参数与 buffer 的状态字典 |
| `optimizer` | AdamW 参数组及历史统计 |
| `model_args` | 重建架构所需参数 |
| `iter_num` | 恢复进度编号 |
| `best_val_loss` | 保存判断使用的值 |
| `config` | 日志与实验配置快照 |

没有保存 GradScaler 状态、Python/NumPy/PyTorch 随机数状态、已经预取的下一批数据，也没有把 `meta.pkl` 嵌入 checkpoint。因此 resume 能继续学习，但不保证与不中断运行逐位一致。

### 13.6 eval_only 的实际条件

退出条件写的是 `if iter_num == 0 and eval_only`。从 scratch 的第 0 步可以只评估退出；若恢复到一个大于 0 的迭代计数，不能仅凭 `--eval_only=True` 就断言这份代码不会继续训练。

这是阅读源码时要养成的习惯：参数名表达意图，判断语句决定实际行为。

### 13.7 当前环境的 checkpoint 加载兼容性

本地 `train.py` 和 `sample.py` 都调用 `torch.load(ckpt_path, map_location=device)`，没有显式指定 `weights_only`。已有项目指南提醒过可能遇到 `Weights only load failed`。

这份 checkpoint 不只有裸权重，`best_val_loss` 还可能是由评估产生的标量张量。本文没有用真实训练产物测试完整保存/恢复链路，不把兼容性风险写成已发生的故障。真遇到问题时，应结合完整异常与本机加载实现定位；只对自己生成并信任的文件考虑指南里的 `weights_only=False` 方案。

## 14. 生成文本的完整路径

阅读 `sample.py` 和 `model.py` 第 306 行的 `generate()`。

### 14.1 sample.py 是外层胶水

它先读配置并设置随机种子，加载 checkpoint，按 `model_args` 重建模型，加载参数，调用 `eval()` 并移动到设备。

随后根据 checkpoint 的 `config['dataset']` 查找 `data/<dataset>/meta.pkl`。找到后使用字符表编码；找不到则假设 GPT-2 BPE。

**对本次字符模型，meta 丢失不能靠 BPE 回退解决。** 编码得到的 ID 可能远超 65 的词表范围，导致索引越界或错误语义。

提示 `ROMEO:` 编码后是长度为 6 的 ID 列表，再加 batch 维变成 `[1,6]`。`FILE:路径` 则表示从文件读取提示。当前实现需要非空、可编码的提示。

### 14.2 generate 每次只添加一个 token

```python
for _ in range(max_new_tokens):
    idx_cond = 最近不超过 block_size 个 token
    logits, _ = self(idx_cond)
    logits = logits[:, -1, :] / temperature
    过滤 top_k 之外的候选
    probs = softmax(logits)
    idx_next = multinomial(probs, num_samples=1)
    idx = cat((idx, idx_next), dim=1)
```

第一轮输入 `ROMEO:`，采样一个字符，例如换行；第二轮把这个新字符也作为输入的一部分，再预测后续字符。训练阶段有真实后续字符作标签，生成阶段后续输入来自自己的采样结果。

训练可以并行计算所有已知位置，生成有这种逐 token 的依赖，所以不能把完整生成过程简单合成一次普通前向。

### 14.3 temperature 与 top_k

temperature 除在 logits 上。小于 1 通常使概率更集中，大于 1 使分布更平缓。这里没有对 0 作特殊处理，不能通过 `temperature=0` 表示贪心解码，应使用正值。

top_k 找到第 k 大分数，将更小的值设为负无穷。代码用 `min(top_k,V)` 防止 k 超过词表大小，所以本项目默认 `top_k=200` 面对 V=65 时不会实际截掉候选。若临界分数并列，保留数可能超过 k。

最后 `torch.multinomial()` 按概率抽样，而不是直接取最大值。`top_k=1` 在没有最大值并列时只保留最高分候选。

### 14.4 上下文长度不等于输出总长度

当 idx 超过 128 个 token，模型只看到最后 128 个，但 idx 保留完整历史，因此最终输出仍能超过 128 个字符。它不能据此关注任意早期文本。

每次被截取的窗口都会重新使用 0—127 的绝对位置编号。该实现没有 KV cache，每次都重新计算窗口主干；也没有按结束 token 自动终止的逻辑，而是循环固定的 `max_new_tokens` 次。

返回的序列包含原始提示。`num_samples` 在外层逐次运行，并非自动拼成一个多样本 batch；随机种子只在开头设置一次，所以多次采样会继续消耗随机数，通常得到不同结果。

## 15. 多卡、编译与性能代码

这些不是第一次读懂单卡训练的前提，但需要知道它们的作用范围。

### 15.1 DDP 做数据并行

每个进程持有一份完整模型，处理自己的随机样本，通过梯度同步协作更新。这里不是把 4 层模型拆到 4 张卡上。

`RANK` 标识全局进程，`LOCAL_RANK` 用于选择当前机器 GPU，`WORLD_SIZE` 是进程总数。主进程负责主要日志与保存。

特别注意这份代码先执行：

```python
gradient_accumulation_steps //= ddp_world_size
```

若命令行 A=8、world size=4，每卡实际累积 2 次；总 token 为 `2*4*B*T=8*B*T`。不能又把原始 A=8 乘 world size 得出 32 倍。

中间微批次不做梯度同步，最后一次才同步，以减少通信。原生 Windows 单卡实验不需要进入默认 NCCL 的多卡路径。

### 15.2 torch.compile

它在模型构造、优化器创建之后包装模型，尝试优化执行。可能有启动编译成本，因此源码在日志里提示首次较慢。你的指南关闭 compile，使第一次实验主要关注模型与训练流程。

编译没有改变字符表，也没有自动把小模型变成更大模型。状态字典里 `_orig_mod.` 前缀的清理代码与这层包装有关。

### 15.3 MFU 不是任务管理器的 GPU 利用率

`estimate_mfu()` 以模型结构估计 FLOPs，除以一轮时间，再除以源码硬编码的 `312e12`，即其采用的 A100 BF16 峰值基准。

因此 RTX 3050 上的 mfu 不能解释为“本机显卡用了百分之几”。它不是本机峰值校准结果，也不是硬件计数器实测值。开头负值还是尚未形成估计的占位状态。

### 15.4 bench.py 的用途

它是性能基准脚本，减少部分训练管理逻辑，可使用真实数据或固定随机输入，并提供 profiler 分支。它实际会进行反向与参数更新，不是纯只读性能查询。

本地 bench 的模型层数、头数、隐藏维度在构造处写成 12/12/768，默认还读 OpenWebText。不要因为文件短就把它当作适合当前字符小模型的无害检查入口。

## 16. 把源码对应到你的机器

### 16.1 每一个小配置控制哪部分成本

| 参数 | 改变什么 | 对计算/显存的主要影响 |
|---|---|---|
| `batch_size` | 同时输入多少窗口 | 激活与计算通常随 B 增长 |
| `block_size` | 每个窗口多少 token | 激活增长；手写注意力分数矩阵随 T² 增长 |
| `n_embd` | 特征宽度 | 主干参数与线性计算中有大量 C² 项 |
| `n_layer` | Block 数 | 参数与主干计算大致随 L 增长 |
| `n_head` | C 如何分配给各头 | 固定 C 时主要改变注意力划分，投影参数总量不因此成倍增加 |
| `gradient_accumulation_steps` | 更新前累积多少批 | 增加每次更新工作量，不等于同时放入更多窗口 |
| `eval_iters` | 验证平均的批次数 | 影响评估耗时与抽样波动 |
| `max_iters` | 循环退出阈值 | 控制实验总长度，注意前述边界 |

高效注意力接口未必显式保存完整 `[B,H,T,T]` 矩阵，因此不能仅按手写分支的张量推算本机实际峰值显存。

### 16.2 原版 Shakespeare 配置并不是指南的小配置

原文件是 6 层、6 头、384 隐藏维、T=256、B=64。指南通过命令行覆盖为 4 层、4 头、128 隐藏维、T=128、B=4。

如果只执行 `train.py config/train_shakespeare_char.py`，就没有启用这些本机缩小设置。阅读实验日志时应核对实际覆盖参数，而不是仅看文件名。

### 16.3 观察训练时，怎样把现象追到源码

| 现象 | 优先看哪里 | 先区分什么 |
|---|---|---|
| 找不到数据文件 | `train.py:get_batch` 和工作目录 | 路径错误还是尚未准备数据 |
| embedding 索引越界 | tokenizer、meta、GPTConfig | ID 范围是否与词表匹配 |
| 显存不足 | B/T/C/L 与实际设备状态 | 参数状态成本还是激活成本 |
| loss 抖动 | 日志出处、抽样、dropout | 最后微批次值还是评估均值 |
| 没有 ckpt.pt | 保存条件与 eval_interval | 是否只有 step 0、是否改善 |
| 恢复结果不完全一致 | checkpoint 字段与随机状态 | 继续训练还是逐位重放 |
| 输出不理解中文 | 字符表与训练目标 | 是否有中文字符、是否做过指令训练 |
| mfu 数字很小 | `estimate_mfu` | 是否误用了 A100 参照解释本机 |

这些是定位路线，不表示本机已经发生对应故障。

### 16.4 这个模型到底学到什么

模型学习给定前缀后，下一个字符的条件分布。训练有效时，生成可能逐渐具备英文词形、人物名和戏剧对话格式。

训练目标里没有系统消息、用户角色、助手回答，也没有“回答问题必须正确”的任务约束。把文本续写模型训练成功，与获得能对话的指令模型，是两个不同的验收目标。

项目第二步对现成 Qwen 做 LoRA，是另一条模型与数据路径，不是把当前字符 nanoGPT 的权重直接接入 Qwen。

## 17. 自测题与建议阅读顺序

### 17.1 分五轮阅读

**第一轮：只追数据。** 打开 prepare 和 get_batch，自己用 `hello` 写出 X、Y，说明每个位置看什么、预测什么。

**第二轮：只追形状。** 从 GPT.forward 开始，标出 `[B,T] → [B,T,C] → [B,T,V]`，再在注意力内补上 `[B,H,T,D]` 与 `[B,H,T,T]`。

**第三轮：只追参数。** 找到所有 Embedding、Linear、LayerNorm，区分参数与中间激活，说明共享权重为什么只算一份。

**第四轮：只追一次更新。** 从训练 while 循环开始，区分前向、反向、梯度累积、裁剪、参数更新、清空梯度。

**第五轮：追实验生命周期。** 看初始化、验证、保存、恢复，再从 sample.py 走到 generate，解释保存文件与字符表怎样共同恢复生成能力。

### 17.2 自测与答案

| 问题 | 答案 |
|---|---|
| 为什么训练输入中已经存在后面的字符，却不算偷看答案？ | 因果 mask 不允许当前位置访问未来输入位置。 |
| 一个 `[4,128]` batch 只预测 4 个字符吗？ | 训练时提供 512 个下一个字符预测目标。 |
| 输入的 128 与隐藏维的 128 是同一件事吗？ | 前者是时间长度 T，后者是特征宽度 C，只是本例恰好相等。 |
| 多头是否把 128 个字符平均分成 4 段？ | 不会；每个头覆盖全部 128 个位置，拆的是特征维。 |
| backward 后权重已经更新了吗？ | 尚未；它计算梯度，optimizer step 才尝试更新参数。 |
| 累积 4 次时，为什么不每次清空梯度？ | 需要把 4 个微批次的梯度加起来。 |
| eval 模式下还能算 loss 吗？ | 能，传入 targets 即可；模式和 targets 是不同开关。 |
| block_size=128 是否最多生成 128 个字符？ | 不是，它限制每轮可见上下文，输出能更长。 |
| sample.py 不传 targets，为什么仍计算整个前缀？ | 主干需要前缀上下文；优化只发生在最后输出投影处。 |
| checkpoint 是否包含字符表？ | 不包含，需要保留正确的 meta.pkl。 |
| max_iters=100 是否刚好运行 100 次更新循环？ | 本地实现从 0 开始并用大于号终止，会运行 101 次。 |
| 有没有办法仅凭 val loss 判断能否聊天？ | 没有；字符续写损失衡量的不是指令对话能力。 |

### 17.3 不启动训练也能做的阅读练习

在纸上或交互式 Python 中用极小张量验证下面四点即可，不必跑长训练：

1. 构造一个小 GPT，传 X、Y，查看 logits 与 loss 形状；不传 Y，再比较形状。
2. 在 eval 模式下，仅改变输入后半段，验证前半段位置的 logits 不变，以理解因果性。
3. 查看 embedding 与 lm_head 的权重是否是同一个对象。
4. 比较完整参数总数和 `get_num_params()`，验证差值恰好等于位置 embedding 参数数。

如果要尝试实际训练，沿用项目已有预训练指南的资源预检与冒烟流程；本文本身不触发训练。

## 18. 关键位置索引与核验说明

| 文件 | 当前起始行 | 内容 |
|---|---:|---|
| `data/shakespeare_char/prepare.py` | 24 | 词表构建 |
| 同上 | 38 | 数据切分 |
| 同上 | 49 | uint16 落盘 |
| `configurator.py` | 20 | 参数顺序解析 |
| `model.py` | 18 | LayerNorm |
| 同上 | 29 | CausalSelfAttention |
| 同上 | 78 | MLP |
| 同上 | 94 | Block |
| 同上 | 109 | GPTConfig |
| 同上 | 120 | GPT 组件、共享权重与初始化 |
| 同上 | 150 | 参数计数 |
| 同上 | 170 | forward 与 loss |
| 同上 | 195 | 裁剪上下文 |
| 同上 | 207 | 加载 GPT-2 权重 |
| 同上 | 263 | AdamW 参数分组 |
| 同上 | 289 | MFU 估计 |
| 同上 | 306 | 自回归生成 |
| `train.py` | 76 | 应用配置 |
| 同上 | 82 | DDP 和设备初始化 |
| 同上 | 116 | get_batch |
| 同上 | 149 | scratch / resume / gpt2 分支 |
| 同上 | 196 | GradScaler 与 optimizer |
| 同上 | 216 | estimate_loss |
| 同上 | 231 | 学习率计算 |
| 同上 | 255 | 主训练循环 |
| 同上 | 263 | 验证与保存 |
| 同上 | 292 | 微批次累积 |
| 同上 | 320 | 日志与循环终止 |
| `sample.py` | 35 | checkpoint 加载 |
| 同上 | 56 | tokenizer 选择 |
| 同上 | 77 | 提示编码与生成 |

本文依据本地源代码逐段核对，并按完成前验证流程执行了小规模 CPU 前向检查（无反向传播、无优化器更新）：

- 用本文的 4 层配置确认完整参数数为 812,288，默认计数为 795,904。
- 确认 token embedding 与 lm_head 引用同一个参数对象。
- 用 `[2,8]` 的短输入确认有标签时 logits 为 `[2,8,65]`、loss 为标量；无标签时 logits 为 `[2,1,65]`、loss 为 None。
- 在 eval 模式下只改变输入后 4 个位置，确认前 4 个位置 logits 在浮点容差内保持一致。
- 核对本地提交号，并检查文档指向的本地文件存在、代码围栏成对。

以上检查已通过。这些验证不证明正式 GPU 训练、checkpoint 恢复或模型生成质量已经通过验收。本文没有修改 nanoGPT 源码。
