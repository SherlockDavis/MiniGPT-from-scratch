# Part 1：从零搭一个 GPT —— 架构篇

> 系列三篇之一。本篇对应 `model.py`（< 200 行），讲清楚一个 ~30M 参数、能写连贯英文短故事的 GPT 是怎么由五个简单模块组装起来的。
>
> Part 2 → 训练；Part 3 → 文本生成。

## 0. 设计目标

在动手之前先把约束说清楚，省得后面纠结：

- **不依赖 `transformers`**：tokenizer 除外，且只允许 `GPT2Tokenizer`。
- **手写注意力**：禁用 `nn.MultiheadAttention` 和 `F.scaled_dot_product_attention`——目的是把每一步的张量形状都暴露出来。
- **核心模型 < 200 行**：所有抽象都要为这个预算服务。
- **~30M 参数**：在 Colab 免费 T4 上 ~1 小时收敛到 val PPL < 5。其中 token embedding 就占了约 19M（vocab=50257 × n_embd=384），真正的 transformer 主体只有 ~10M。

最终输入到输出的数据流：

```
Input tokens (B, T)
  → Token Embedding + Position Embedding   (B, T, C)
  → 6 × TransformerBlock (Pre-LN)          (B, T, C)
  → final LayerNorm                        (B, T, C)
  → LM Head                                (B, T, vocab_size)
```

`B` = batch，`T` = 序列长度（≤ 256），`C` = `n_embd` = 384。

---

## 1. LayerNorm —— 最基础的稳定剂

为什么要先讲 LayerNorm？因为 Pre-LN 架构里它出现的频率比注意力还高（每个 Block 两次，最后再加一次收尾的 `ln_f`）。

它做的事很简单：**把每个 token 的特征向量拉到均值 0、方差 1，再用一对可学习的 `weight`/`bias` 重新缩放**。

```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        if self.bias is not None:
            return self.weight * x_hat + self.bias
        return self.weight * x_hat
```

几个容易踩坑的细节：

- **`unbiased=False`**：这是有偏方差（除以 `N`），匹配 `torch.nn.LayerNorm` 的行为。`torch.var` 默认是无偏（除以 `N-1`），写错了数值会和官方实现对不上，单测会挂。
- **只在最后一维归一化**：和 BatchNorm 不同——BatchNorm 跨 batch，对小 batch 不稳定；LayerNorm 只看自己这条样本。
- **`eps`** 防止除零，必须加在 sqrt 里面。

> **Pre-LN vs Post-LN**：原始 Transformer 论文是 Post-LN（`x = LN(x + Sublayer(x))`），但深层网络容易梯度爆炸。GPT-2 改用 Pre-LN（`x = x + Sublayer(LN(x))`），残差路径上没有 LN，梯度可以一路畅通到底层。这个差别看起来微小，决定了 6 层以上能不能稳定训练。

---

## 2. CausalSelfAttention —— 整个项目的核心

这是唯一一个真值得讲细节的模块。其它都是包装。

注意力机制的本质是：**让序列里每个位置去 "查询" 其它位置，按相关度加权聚合信息**。每个位置都有三种身份：
- **Q（query）**：我要找什么
- **K（key）**：我能被什么样的查询找到
- **V（value）**：找到我之后能拿走的内容

### 2.1 三件事情：投影、分头、缩放点积

```python
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
self.c_proj = nn.Linear(config.n_embd, config.n_embd)
```

第一处优化：`Q/K/V` 三个投影合并成一个 `(C → 3C)` 的 `Linear`。一次 matmul 在 GPU 上比三次更快，参数量完全一样。

```python
B, T, C = x.shape
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
# 分头：(B, T, C) → (B, T, n_head, head_size) → (B, n_head, T, head_size)
q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
```

`n_embd=384, n_head=6` → `head_size=64`。**多头 = 把 384 维的特征切成 6 段，每段独立做一次 attention**，最后拼回来。这样不同的头可以学到不同类型的依赖（句法 vs 语义 vs 位置），代价只是一次 reshape。

### 2.2 缩放点积

```python
att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)  # (B, n_head, T, T)
```

为什么要除 `sqrt(head_size)`？因为 `q · k` 是 `head_size` 个独立项的求和，方差随维度线性增长。如果不缩放，softmax 输入会变得很大，输出趋近 one-hot，梯度几乎为零——这就是著名的"注意力梯度消失"。除 `sqrt(head_size)` 后方差回到 1 量级，softmax 才能产生有用的梯度。

### 2.3 Causal Mask

GPT 是**自回归**模型——位置 `t` 只能看到 `[0, t]`，绝对不能偷看未来。怎么实现？

```python
# 在 __init__ 里：
mask = torch.tril(torch.ones(config.block_size, config.block_size))
self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

# 在 forward 里：
att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
att = F.softmax(att, dim=-1)
```

三个关键点：

1. **mask 加在 softmax *之前***，用 `-inf` 填充上三角。`exp(-inf) = 0`，softmax 后这些位置权重严格为 0。如果反过来在 softmax 之后再 mask 再 renormalize，数值不严格相等而且更容易出 NaN。
2. **`register_buffer` 而不是 `Parameter`**：mask 不需要学习，但要随 `.to(device)` 跟着走。Buffer 正好是这种东西。
3. **`[:, :, :T, :T]`**：mask 是按 `block_size=256` 预分配的，但实际序列可能更短（比如生成时 prompt 只有 5 个 token），所以要切片。

形状广播一定要看清楚：mask 是 `(1, 1, block_size, block_size)`，`att` 是 `(B, n_head, T, T)`。两个 `1` 维度自动广播过去，一份 mask 给所有 batch、所有 head 共用。

### 2.4 输出回到原维度

```python
y = att @ v                                          # (B, n_head, T, head_size)
y = y.transpose(1, 2).contiguous().view(B, T, C)     # 合并头
y = self.resid_dropout(self.c_proj(y))               # 输出投影 + dropout
```

**`.contiguous()` 是必须的**——`transpose` 之后内存不连续，直接 `.view()` 会报错。换成 `.reshape()` 也行，但显式 `.contiguous().view()` 让内存意图更清楚。

### 2.5 两个 Dropout，位置不一样

```python
self.attn_dropout = nn.Dropout(config.dropout)   # 加在 softmax 后的注意力权重上
self.resid_dropout = nn.Dropout(config.dropout)  # 加在输出投影后、回到残差流前
```

第一个 dropout **随机扔掉一些注意力连接**（softmax 之后那张 (T, T) 表里的若干元素清零），第二个 dropout **干扰整个 sublayer 的输出**。两者作用层级不同，不能合并。

---

## 3. MLP —— 廉价但不可或缺

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))
```

每个 token 独立过一个两层 MLP（`C → 4C → C`）。两个细节：

- **隐藏维度 `4*C`**：原始 Transformer 论文就是 4 倍，沿用至今。这一点甚至比注意力本身更"经验主义"。
- **GELU 而不是 ReLU**：GPT-2 / GPT-3 全用 GELU。它在 0 附近平滑过渡，对小信号更友好。

把 attention 看作"信息混合"，MLP 看作"信息加工"——两者交替进行。如果只有 attention 没有 MLP，模型只能做线性组合；如果只有 MLP 没有 attention，token 之间不能通信。

---

## 4. Block —— Pre-LN 残差结构

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # ← 注意先 LN 再 attn，残差在外层
        x = x + self.mlp(self.ln_2(x))
        return x
```

整个 transformer 的脊椎就是反复堆这个结构。两条原则贯穿到底：

1. **Pre-LN**：LN 在 sublayer 之前，残差路径完全干净。
2. **每个 sublayer 都有自己独立的 LN**：不能两个 sublayer 共用一个。

---

## 5. GPT —— 把所有东西串起来

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)   # token embedding
        self.wpe = nn.Embedding(config.block_size, config.n_embd)   # position embedding
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # GPT-2 §2.3: 残差投影的方差按深度缩放
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
```

### 5.1 可学习的 position embedding

```python
self.wpe = nn.Embedding(config.block_size, config.n_embd)   # 256 × 384
```

为什么不用 sinusoidal？因为：
- GPT-2 用的是可学习的，跟它对齐。
- `block_size=256` 很小，参数量只有 `256 × 384 ≈ 0.1M`，便宜。
- 实测在 TinyStories 上两种方案差异不大；可学习的优势是模型可以"任意编码"位置信息。

### 5.2 Weight Tying（权重共享）

```python
self.lm_head.weight = self.wte.weight
```

输入 embedding 和输出 LM head 是**同一份权重**。直觉是：把 token id 投到向量空间（embed），和把向量空间投回 token logits（unembed），本质是互逆操作，参数应该一致。

省掉 `vocab_size × n_embd ≈ 19M` 参数——对一个 30M 的模型来说，这是天大的便宜。

### 5.3 残差投影的特殊初始化

```python
# 普通 Linear：std = 0.02
# 但每个 Block 里 c_proj 的 weight 用更小的：std = 0.02 / sqrt(2 * n_layer)
```

这是 GPT-2 论文的细节。原因：每经过一个 Block，残差流上叠加一份 sublayer 输出，方差线性累积。把残差投影初始化得更小，让网络一开始接近恒等函数，训练更稳。`2 * n_layer` 里的 2 是因为每个 Block 有 2 个 sublayer（attn + mlp），都贡献到残差。

### 5.4 Forward

```python
def forward(self, idx, targets=None):
    B, T = idx.shape
    pos = torch.arange(T, dtype=torch.long, device=idx.device)   # (T,)
    tok_emb = self.wte(idx)                                       # (B, T, C)
    pos_emb = self.wpe(pos)                                       # (T, C) → 广播
    x = self.drop(tok_emb + pos_emb)

    for block in self.blocks:
        x = block(x)
    x = self.ln_f(x)

    logits = self.lm_head(x)                                      # (B, T, V)

    loss = None
    if targets is not None:
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )
    return logits, loss
```

注意几点：

- **`pos_emb` 是 `(T, C)`**，没有 batch 维度，靠广播加到 `tok_emb` 上。
- **`drop` 加在 token+pos 之后、第一层 block 之前**，相当于对输入做 dropout。
- **target 由 caller 提供**，不在 forward 里做左移。这样训练循环可以一次性算 `loss`，generate 时跳过 `targets=None` 不算 loss。
- **`cross_entropy` 自动处理 softmax**：所以 logits 不要先过 softmax 再传进去，会算两次。

---

## 6. 参数量核算

写完之后用 `model.num_parameters()` 验证：

```
Token embedding (wte):    50257 × 384 = 19_298_688   ← 共享给 lm_head
Position embedding (wpe):   256 × 384 =     98_304
6 × Block:
  ln_1 + ln_2:            2 × 384 × 2 =      1_536  (weight + bias)
  c_attn:                 384 × (3×384) + (3×384) = 443_904
  attn.c_proj:            384 × 384 + 384 =  147_840
  mlp.c_fc:               384 × 1536 + 1536 = 591_360
  mlp.c_proj:             1536 × 384 + 384 =  590_208
  ────────────────────────────────────────────
  per block:                                ~1_774_848
  × 6 layers:                              ~10_649_088
ln_f:                     2 × 384 =                768
lm_head:                  共享，0 额外参数
────────────────────────────────────────────
Total:                                    ~30_046_848 ≈ 30.0M
```

token embedding 一个就吃掉了 64% 的预算。这就是为什么大词表的小模型"看起来参数多但其实没多深"。

---

## 7. 我自己踩过的坑

| 坑 | 症状 | 定位 |
|----|------|------|
| `var` 用了无偏估计 | LayerNorm 单测和 `nn.LayerNorm` 差 1e-3 | `unbiased=False` |
| mask 加在 softmax 之后 | loss 一开始就 NaN | 改成 softmax 之前 mask 成 -inf |
| `transpose` 后直接 `.view()` | RuntimeError: view size is not compatible | 加 `.contiguous()` |
| 忘了 weight tying | 参数量 ~50M，OOM | `self.lm_head.weight = self.wte.weight` |
| target 没左移 | loss 不下降，甚至上升 | 检查 `get_batch` 的 `y = data[i+1 : i+1+block]` |
| pos_emb 加错维度 | shape mismatch | `pos = torch.arange(T)`，让广播帮你做 |

---

## 下一步

模型搭好了，下一步是怎么把它训练起来，并且让训练在 Colab T4 上 1 小时跑完。Part 2 会讲：

- 数据流水线（如何把 2GB TinyStories 编码成 `train.bin` / `val.bin`）
- 混合精度训练（FP16 + GradScaler）
- 梯度累积（如何在小显存上模拟大 batch）
- LR schedule（warmup + cosine）
- 训练监控（wandb）

→ 见 [Part 2: 训练篇](./part2_training.md)
