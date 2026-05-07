# Part 2：从零搭一个 GPT —— 训练篇

> 系列三篇之二。本篇对应 `train.py` + `utils.py`，讲清楚怎么把 [Part 1](./part1_architecture.md) 搭好的 30M 模型，在 Colab T4 上 ~1 小时训练到 val PPL ≈ 4.7。
>
> Part 3 → 文本生成。

## 0. 基线先跑通，再加优化

我做这个项目时严格按 CLAUDE.md 里的工作流走：**先把最简单的训练循环跑通 100 步看 loss 下降，再叠加 AMP / 梯度累积 / LR schedule / wandb**。理由很现实——一上来全堆上，出了问题不知道是哪个环节挂了。

```
step 8:  最朴素的 AdamW 循环，常数 lr，无 AMP，无 wandb
step 9:  接 wandb
step 10: 加 AMP + 梯度累积 + cosine schedule
step 11: 完整 10000 步训练
```

下面按"为什么这样设计"展开，所有代码都和 `train.py` 对应。

---

## 1. 数据流水线 —— 慢的部分要先跑一次

最容易被低估的环节。TinyStoriesV2 训练集是个 ~2 GB 的纯文本文件，每个故事用 `<|endoftext|>` 分隔。如果每个 step 才现场 tokenize，训练会被 CPU 拖死。

正确做法：**预处理一次，存成 token id 流**。

### 1.1 选择 dtype：uint16 还是 int64？

```python
TOKEN_DTYPE = np.uint16
```

GPT-2 vocab=50257，最大 token id 是 50256，远小于 `uint16` 的上限 65535。用 `uint16` 比 `int64` 省 **4 倍**磁盘和 I/O。memmap 读这个文件时也快 4 倍。

### 1.2 选择 tokenizer：fast vs slow

```python
def get_tokenizer():        # 慢的 Python 实现，generate.py 用
    return GPT2Tokenizer.from_pretrained("gpt2")

def get_fast_tokenizer():   # Rust 实现，预处理用
    return Tokenizer.from_pretrained("gpt2")
```

两者**词表完全一致**（都来自 GPT-2 官方），但 Rust 版（`tokenizers` 库）有 `encode_batch` 可以并行，而且 BPE 主循环本身就快 10–50 倍。预处理 2GB 文本的差别是"几分钟"vs"几小时"——差到必须用 fast 版。

### 1.3 写出来：分块、不要一次性 list 化

```python
stories = [s for s in (story.strip() for story in text.split(STORY_SEPARATOR)) if s]
del text  # 释放原始文本占的内存

with open(output_path, "wb") as fout:
    for start in chunk_starts:
        batch = stories[start : start + chunk_size]
        buf = []
        for enc in tokenizer.encode_batch(batch):
            buf.extend(enc.ids)
            buf.append(eot_id)        # 每个故事尾巴显式补 EOT
        np.asarray(buf, dtype=TOKEN_DTYPE).tofile(fout)
```

两个细节：

- **每个故事末尾显式追加 `<|endoftext|>` (id=50256)**：让模型有"故事边界"信号，不会把两个故事错连成一段。
- **chunk-by-chunk 写文件**：如果先把所有 token 全部装进 Python list，2GB 文本编码后大约 14GB（Python int 每个 28 字节）。分块后内存占用只有几 MB。

### 1.4 读取：memory-mapped，不进内存

```python
def load_tokens(path):
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")
```

`memmap` 把文件假装成数组，但实际只有被访问的部分会被 OS 拉进 page cache。500MB 的训练 token 文件甚至不用考虑内存。

### 1.5 采样 batch：随机起点 + 左移一位

```python
def get_batch(data, block_size, batch_size, device="cpu"):
    high = len(data) - block_size - 1
    starts = np.random.randint(0, high, size=batch_size)
    x = torch.from_numpy(np.stack([data[i     : i + block_size    ].astype(np.int64) for i in starts]))
    y = torch.from_numpy(np.stack([data[i + 1 : i + 1 + block_size].astype(np.int64) for i in starts]))
    if str(device).startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    return x, y
```

四个关键点：

1. **`y = x` 左移一位**：next-token prediction 的本质。位置 `t` 的 label 是位置 `t+1` 的 token。
2. **随机起点、有放回**：不分 epoch、不洗牌——这是 nanoGPT 风格。在 TinyStories 这个规模上，每条样本平均被见到的次数远小于 1，重复采样不是问题。
3. **`int64`**：`nn.Embedding` 的 index 必须是 long。
4. **`pin_memory + non_blocking`**：把数据放到 page-locked 内存里，CUDA 可以异步 DMA，掩盖部分 H2D 传输时间。

> ⚠️ 把 `y = x` 左移这一步搞错（比如 `y = data[i:i+block]` 写成 `y = data[i:i+block]` 没移位），loss 不会报错但会卡在很高的值不动——这是新手最容易遇到的坑之一。

---

## 2. 朴素训练循环 —— 先确认信号是健康的

```python
def train_step(model, optimizer, x, y, grad_clip):
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item()
```

最简版：单 batch、fp32、AdamW、常数 lr、grad clip。跑 100 步看 loss 从 ~11（即 `ln(50257)`，随机初始化的均匀分布交叉熵）开始下降到 ~7—— 健康的信号。

**`zero_grad(set_to_none=True)`**：把梯度设成 `None` 而不是 0，省一次写操作，PyTorch 推荐做法。

**`grad_clip=1.0`**：把全部参数的梯度二范数 clip 到 1.0。Transformer 训练初期梯度可能突然爆掉，clip 是非常便宜的保险。

---

## 3. 三个加速器：AMP、梯度累积、LR Schedule

确认基线健康后，把三件套加上去。

### 3.1 混合精度（AMP / FP16）

```python
use_amp = device == "cuda"
scaler = torch.amp.GradScaler("cuda") if use_amp else None

with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
    _, loss = model(x, y)
    loss = loss / grad_accum_steps
if use_amp:
    scaler.scale(loss).backward()
else:
    loss.backward()

if use_amp:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    scaler.step(optimizer)
    scaler.update()
```

为什么要这么复杂？因为 FP16 的动态范围是 `~6e-5 到 6.5e4`，**反向时小梯度直接 underflow 成 0**。GradScaler 的对策是：

1. **forward + loss 算完后，把 loss 乘以一个大数（默认 65536）再 backward**——所有梯度跟着放大，避免 underflow。
2. **step 之前把梯度除回去（unscale）**——这样 grad clip 看到的是真实的 grad norm，clip 行为正确。
3. **如果 unscale 之后发现 inf/nan（说明 scaler 设太大）**，自动跳过这一 step 并把 scaler 减半；如果连续 N 步都正常，自动放大 scaler。这个动态自适应是 GradScaler 的精髓。

T4 上 FP16 比 FP32 大约**快 2 倍、显存省 2 倍**——这是能在免费 Colab 上跑 30M 模型的关键。

> ⚠️ `autocast` 只覆盖 forward。loss.backward() 不在 autocast 上下文里——它沿用 forward 时各 op 的实际 dtype。

### 3.2 梯度累积

```python
optimizer.zero_grad(set_to_none=True)
loss_accum = 0.0
for _ in range(grad_accum_steps):
    x, y = get_batch(...)
    with torch.amp.autocast(...):
        _, loss = model(x, y)
        loss = loss / grad_accum_steps   # 关键：除一下
    if use_amp:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    loss_accum += loss.item()
optimizer.step()  # 只在 N 个 micro-batch 累完之后 step 一次
```

直觉：**N 个小 batch 累计的梯度 ≈ 一个大 N 倍 batch 的梯度**（在严格意义上需要 loss 是平均而非求和）。所以：

- `loss = loss / grad_accum_steps`：除一下，这样 N 个 micro-batch 累出来的梯度等于在一个 N×B 的大 batch 上算的平均梯度。如果忘了除，相当于把 lr 偷偷放大了 N 倍。
- **每个 micro-batch 重新 `get_batch`**：用不同的数据，否则就是单个小 batch 算 N 遍——纯浪费算力。

我跑的本次实验里 `grad_accum_steps = 1`（默认），但参数留出来很必要——别人想在 6GB 显存的卡上跑就可以 `--batch-size 16 --grad-accum-steps 4` 保持有效 batch=64。

### 3.3 LR Schedule（warmup + cosine）

```python
def get_lr(step, warmup_iters, max_iters, peak_lr, min_lr):
    if step < warmup_iters:
        return peak_lr * (step + 1) / warmup_iters
    if step >= max_iters:
        return min_lr
    decay_ratio = (step - warmup_iters) / max(1, max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)
```

形状是这样：

```
lr
│        ╱╲
│       ╱   ╲
│      ╱      ╲
│     ╱         ╲___
│    ╱              ‾‾‾___ min_lr
│___╱
└──────────────────────── step
   500           10000
   warmup
```

为什么需要 warmup？训练刚开始时 weights 还没被数据"塑形"，loss 曲面在初始点附近可能很陡，**直接用 peak_lr 容易把模型踢到很奇怪的位置**。先用很小的 lr 让 Adam 把它的二阶矩估计稳定下来，再慢慢爬到 peak_lr。500 步对一个 10000 步的训练来说是 5%。

为什么要 cosine 而不是 linear decay？经验。cosine 在中段下降慢、末尾下降快，给模型更多时间在 peak 附近探索；linear 早期下降太狠。最低值取 `peak_lr * 0.1` 是 GPT-2 / GPT-3 的惯例。

### 3.4 全部串起来的训练循环

```python
for step in range(max_iters):
    # 1) 设 lr
    lr = get_lr(step, config.warmup_iters, max_iters, config.learning_rate, min_lr)
    for g in optimizer.param_groups:
        g["lr"] = lr

    # 2) 梯度累积 (单 step 内)
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    for _ in range(grad_accum_steps):
        x, y = get_batch(train_data, ...)
        with torch.amp.autocast(...):
            _, loss = model(x, y)
            loss = loss / grad_accum_steps
        scaler.scale(loss).backward() if use_amp else loss.backward()
        loss_accum += loss.item()

    # 3) Unscale → clip → step
    if use_amp:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

    # 4) Health check
    if not np.isfinite(loss_accum):
        raise RuntimeError(f"Non-finite loss at step {step}")
```

一段循环，五个事情：set lr → 累积梯度 → unscale + clip + step → 监控 → fail-fast。**`if not np.isfinite(loss): raise`** 是关键——FP16 训练偶尔会出 NaN，必须让训练立刻挂掉，不要让一个坏 step 污染后面所有的 weights。

---

## 4. 评估和保存

```python
@torch.no_grad()
def estimate_loss(model, train_data, val_data, ..., eval_iters=20):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, ...)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out
```

为什么不用整个 val set？因为：
1. **每 250 步评估一次**，整个 val set 太慢。
2. **20 个随机 batch 的平均已经足够稳定**（噪声水平远小于 train/val 的真实 gap）。
3. 真正的最终评估留到训练结束后再做一次。

记得把 model 切到 eval 模式（关闭 dropout）然后切回 train 模式，否则下面的训练会一直在 dropout 关闭状态。

Checkpoint 包括 model + optimizer + config + step：

```python
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "config": config,    # 整个 GPTConfig dataclass
    "step": step,
}, path)
```

为什么要存 optimizer？因为 AdamW 维护一阶和二阶动量，如果中途断电要续训，没有 optimizer 状态就要从头预热。

---

## 5. 监控：logger + 可选 wandb

项目规则禁止 `print` 调试，所以两条监控通道：

**默认通道：Python `logging`**——所有 step / loss / lr / eval 都通过 `logger.info(...)` 写到 stdout 和文件，重定向后就是完整训练日志（本仓库的 `step11.log` 就是这么来的，README 里的 loss 曲线也是从它解析出来的）。

**可选通道：wandb**——`train.py` 默认开启（`use_wandb=True`），传 `--no-wandb` 关闭。如果开启，需要先 `wandb login`：

```python
if use_wandb:
    wandb.init(project="minigpt", name=run_name, config={**asdict(config), ...})

# 训练循环里：
wandb.log({"train/loss": loss_val, "train/lr": lr}, step=step)
wandb.log({"eval/train_loss": ..., "eval/val_loss": ...}, step=step)
```

为什么 wandb 比 logger 强？
- logger 是写完即焚，wandb 自动记录 GPU 利用率、显存、wall clock，免费帮你做归档。
- 多次实验对比时可以叠图。
- 但如果你只跑一次或者不想注册账号，纯 logger + 一个解析脚本（见 `scripts/plot_loss_curve.py`）也完全够用。

这是我跑完 10000 步看到的曲线趋势：

```
step 0     | loss 11.0    (≈ ln(50257)，随机初始化基线)
step 500   | loss 5.5     (warmup 结束)
step 1000  | loss 3.8
step 5000  | loss 2.1
step 9999  | loss 1.59    | val loss 1.55  → PPL ≈ 4.7
```

83 分钟，Colab 免费 T4，每 step 约 0.5 秒。

---

## 6. 我自己踩过的坑

| 坑 | 症状 | 定位 |
|----|------|------|
| 用 `GPT2Tokenizer` 预处理整个 train.txt | 跑了 3 小时还在 tokenize | 换 `tokenizers.Tokenizer`（fast 版） |
| 忘记 `loss / grad_accum_steps` | loss 爆炸/不下降，相当于 lr 偷偷 ×N | 除一下 |
| AMP 下 `clip_grad_norm` 在 unscale 之前 | grad clip 看到的是放大后的梯度，等于没 clip | 先 `scaler.unscale_(optimizer)` 再 clip |
| 评估完忘了 `model.train()` | 训练时 dropout 一直关着，过拟合速度异常快 | 评估函数收尾切回 train 模式 |
| token 文件用 int64 | I/O 慢 4 倍，磁盘也吃满 | uint16 |
| `pos_emb` 没广播到 batch | shape mismatch | `pos = torch.arange(T)`，依赖广播 |
| 数据预处理写成一次性 list | OOM | 分块写盘 |

---

## 7. 复现命令

```bash
# 一次性数据预处理（首次训练自动跑）
bash scripts/download_data.sh

# 跑 100 步烟雾测试
python train.py --max-iters 100 --no-wandb

# 完整训练（10000 步，~1 小时 T4）
python train.py

# 显存只有 6GB？保持有效 batch=64
python train.py --batch-size 16 --grad-accum-steps 4
```

---

## 下一步

模型训练完毕，checkpoint 存好了。下一篇讲怎么从 logits 采样出文本——四种解码模式（greedy / temperature / top-k / top-p），每种适合什么场景，附本仓库实测样本。

→ 见 [Part 3: 文本生成篇](./part3_generation.md)
