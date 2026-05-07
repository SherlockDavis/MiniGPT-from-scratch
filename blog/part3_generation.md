# Part 3：从零搭一个 GPT —— 文本生成篇

> 系列三篇之三。本篇对应 `generate.py`，讲清楚怎么从训练好的 [Part 1](./part1_architecture.md) 模型 + [Part 2](./part2_training.md) 的 checkpoint 里采样出文本，以及四种解码模式各自适合什么场景。
>
> 文末附本仓库 `ckpt_step10000.pt` 的实测样本。

## 0. 解码不是"反向训练"

很多人第一次接触 GPT 会以为：训练好的模型 forward 一次就给出最终文本。其实不是——模型只能预测**下一个** token 的概率分布，剩下都靠**自回归循环**：

```
prompt = "Once upon a time"
for i in range(max_new_tokens):
    logits = model(prompt)[-1]              # 取最后一个位置的 logits
    next_token = sample_from(logits)        # 从分布里挑一个
    prompt = prompt + [next_token]          # 拼回去
```

整个生成的成本 = `max_new_tokens` 次 forward。每次 forward 又从头算一遍——是的，效率很差，工业级实现会用 KV cache，本仓库为了简单不做这个优化。

四种解码模式，本质都是**怎么从 logits 里挑下一个 token**：

| 模式 | 怎么挑 | 适合场景 |
|------|--------|----------|
| Greedy | argmax，永远挑概率最高的 | 复现性、调试、抽签 |
| Temperature | 把分布整体变平/变尖再 sample | 控制"创造性" |
| Top-k | 只在 k 个最高概率里 sample | 排除 long-tail 噪声 |
| Top-p (nucleus) | 在累计概率 ≥ p 的最小集合里 sample | 自适应"长尾"边界 |

---

## 1. 主循环

```python
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
    block_size = model.config.block_size
    for _ in range(max_new_tokens):
        # context cropping：模型只能看 block_size 内
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]   # (B, V) — 只要最后一个位置的分布

        if temperature == 0.0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                logits = _apply_top_k(logits, top_k)
            if top_p is not None:
                logits = _apply_top_p(logits, top_p)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_token], dim=1)
    return idx
```

几个关键设计：

- **`@torch.no_grad()`**：generate 不需要梯度，关掉省显存。
- **context cropping**：`idx[:, -block_size:]` 永远只把最后 `block_size` 个 token 喂给模型——超过这个长度位置 embedding 就没定义了。代价是模型"记不住"超出 block_size 的开头。
- **`logits[:, -1, :]`**：模型 forward 出来是 `(B, T, V)`，但生成时只关心**最后一个**位置的预测（前面那些位置的预测在前一步循环里已经用过了）。
- **`temperature == 0.0` 走单独的 argmax 分支**：因为 `logits / 0` 是除零异常，必须在数值上短路掉。
- **`torch.multinomial(probs, num_samples=1)`**：从分类分布里抽 1 个样本。比手写 `torch.cumsum + searchsorted` 干净得多。

---

## 2. Greedy（temperature == 0）

```python
next_token = torch.argmax(logits, dim=-1, keepdim=True)
```

最朴素的策略：每一步都挑概率最高的 token。

**优点**：
- 完全确定性，相同输入永远相同输出。调试模型时是必须用的。
- 在 well-trained 模型上往往给出最"流畅"的输出。

**缺点**：
- 容易陷入"安全但乏味"的循环。模型一旦发现某个 phrase 概率高，会反复使用。
- 对 prompt 几乎没有变化空间。

什么时候用：写单测、算 BLEU、确认模型真的学到了什么、抽签做演示。

---

## 3. Temperature

```python
logits = logits / temperature
probs = torch.softmax(logits, dim=-1)
```

把所有 logits 除一个温度 `T`，然后 softmax。

数学上的效果：
- `T = 1.0`：原始分布，不变。
- `T > 1.0`（比如 1.5、2.0）：分布被**拉平**——高概率 token 概率下降，低概率 token 概率上升。生成更"野"。
- `T < 1.0`（比如 0.7、0.5）：分布被**变尖**——高概率 token 越发突出。生成更"保守"。
- `T → 0`：极限就是 argmax（greedy）。
- `T → ∞`：极限就是均匀分布（纯随机）。

**经验区间**：0.7–0.9 在故事生成里通常是"流畅 + 有变化"的甜蜜点。我在 README 里展示的样本用的 0.8。

> 注意 `T` 是除在 logits 上，不是除在 probs 上。除在 probs 上之后再 normalize 数学不等价（不是单调变换），结果会很奇怪。

---

## 4. Top-k

```python
def _apply_top_k(logits, top_k):
    k = min(top_k, logits.size(-1))
    threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))
```

只保留 logits 最高的 `k` 个 token，其它全部 `-inf`（softmax 后概率为 0）。

实现细节：
- `torch.topk(logits, k).values` 是 `(B, k)`，按降序排列。
- `[..., -1, None]` 取每行的第 k 大值（即"门槛"），`None` 加一个维度方便广播。
- `masked_fill(< threshold, -inf)`：低于门槛的全部置 `-inf`。

**为什么需要 top-k**：

GPT-2 vocab 是 50257。即使训练得很好，**长尾噪声永远存在**——某个完全不相干的 token 可能因为模型不够确信而被分到 1e-5 的概率。50257 × 1e-5 = 0.5，加起来这个"长尾"在 multinomial sample 里有 50% 概率被抽到。一旦抽到一个怪 token，后续生成就跑偏了。

**top-k 的局限**：

`k` 是**固定**的，但有些场景模型其实非常确信（前 1 个就吃掉 95%），有些场景模型很不确信（top 100 才吃 80%）。固定 `k` 在前者会"放进太多噪声"，在后者会"砍得太狠"。

**经验值**：`k = 40` 是 GPT-2 论文用过的值，效果可以。

---

## 5. Top-p（nucleus）—— 自适应版的 top-k

```python
def _apply_top_p(logits, top_p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_to_remove = cumulative_probs > top_p
    sorted_to_remove[..., 1:] = sorted_to_remove[..., :-1].clone()
    sorted_to_remove[..., 0] = False

    to_remove = sorted_to_remove.scatter(-1, sorted_indices, sorted_to_remove)
    return logits.masked_fill(to_remove, float("-inf"))
```

思路：**从高到低累计概率，找到第一个让累计 ≥ p 的位置 k\*，只保留前 k\* 个**。

举两个极端：
- 模型很确信 → top 1 就 95%，`p=0.9` → 只保留 1 个 token。
- 模型很不确信 → 要 top 100 才达到 80%，`p=0.9` → 保留 ~100 个。

这就是它叫 "nucleus"（核）的原因——动态地划出概率密度的"核心"。

实现里有几个微妙的点：

### 5.1 为什么要先排序

要计算"累计概率 ≥ p 的最小集合"必须先按概率从高到低排。`torch.sort(descending=True)` 同时返回 `sorted_logits` 和 `sorted_indices`（原 vocab 里的位置）。

### 5.2 "向右移一位"的处理

```python
sorted_to_remove = cumulative_probs > top_p          # 累计 > p 的就要被剔除
sorted_to_remove[..., 1:] = sorted_to_remove[..., :-1].clone()   # 向右移 1
sorted_to_remove[..., 0] = False                                  # 第 1 个永远保留
```

为什么要移一位？想象 `p=0.5`，第一个 token 概率就 0.7。直接按 `cumulative > 0.5` 标记，第一个 token 就被剔除了，**整个分布全空**。

正确语义是"保留**让累计第一次 ≥ p 的那个 token**"——它本身要保留，再下一个才剔除。所以把 mask 整体右移一位。

### 5.3 把 sorted-position 的 mask 还原回 vocab-position

```python
to_remove = sorted_to_remove.scatter(-1, sorted_indices, sorted_to_remove)
```

`sorted_to_remove` 是按 sorted 顺序的，但要应用到原始 logits 上必须按原 vocab 顺序。`scatter` 把 sorted 索引位置的值散射回原位。

> 这个反向映射是 top-p 实现里最容易搞错的地方。一种替代写法是直接把 sorted_logits mask 之后再用 `scatter` 写回原序，最终效果一样。

### 5.4 经验值

`p = 0.9` 是默认推荐。低于 0.85 会变得过于保守，高于 0.95 长尾噪声开始回来。

---

## 6. 组合

可以同时启用 `top_k` 和 `top_p`：

```python
if top_k is not None:
    logits = _apply_top_k(logits, top_k)
if top_p is not None:
    logits = _apply_top_p(logits, top_p)
```

执行顺序是先 top-k 后 top-p，效果是**取两个集合的交集**。常见组合是 `top_k=40, top_p=0.9`——既限制绝对数量上限，又允许长尾自适应收紧。

注意：**top-k/top-p 都不会改变保留下来的 token 之间的相对概率**（只是把其它的设为 0）。最终 sample 仍然是从重新归一化的子分布里抽。

---

## 7. 实测：本仓库 step10000 checkpoint

prompt 一律为 `"Once upon a time"`，max_new_tokens=200，seed=42。

### Greedy (`--temperature 0.0`)

> Once upon a time, there was a little girl named Lily. She loved to play with her toys and eat yummy food. One day, she found a big, red apple in her kitchen. She was very happy and wanted to eat it all.
>
> Lily tried to pick the apple, but it was too high. She tried to jump, but she could not reach it. She felt sad and sat down on a chair. A kind bird saw Lily and asked, *"Why are you sad, Lily?"*
>
> Lily said, *"I want the apple, but it is too high for me."* The bird had an idea. The bird flew up and picked the apple for Lily. Lily was so happy and thanked the bird. […] And they lived happily ever after. `<|endoftext|>`

完整三段式叙事，命名一致，最后自然出现 EOT。这种"教科书式"输出是 greedy 的典型——所有概率最高的选择串起来形成最"中规中矩"的故事。

### Temperature 0.8

> Once upon a time, there was a little boy named Tim. Tim had a friend, a big cat named Sam. They liked to play together. One day, Tim and Sam wanted to make a game with the chalk.
>
> Tim said, *"Let's play!"* Sam agreed, and they started to play. They drew a big sun, a house, and a tall tree. They laughed and had fun. […] `<|endoftext|>`

注意主角换了人（Tim+Sam，不是 greedy 的 Lily）——同一个 prompt 同一个 seed，换了解码模式输出完全不同。这是采样的本质。

### Top-k 40 (`--temperature 1.0 --top-k 40`)

> Once upon a time, there was a little boy named Tim. Tim had a friend, a big cat named Sam. They liked to play all day. One day, they found a big, pretty curtain in the living room. They were very curious about what was inside. […]

故事情节比 temperature 0.8 更野（curtain → hide and seek），但偶尔有小语病（"started to count, 'One, two...'"）。

### Top-p 0.9 (`--temperature 1.0 --top-p 0.9`)

> Once upon a time, there was a little boy named Tim. Tim had a friend named Sam. They liked to play with their toys and have fun together. One day, they wanted to make a big igloo. […]

物理逻辑出现了一些跳跃（"put out the fire and tried to put it in the igloo"）——这是采样自由度高时的典型副作用。

### 已知短板：复读

```
prompt = "The little dragon was scared because"
解码 = top-p 0.9 + temperature 0.8

输出节选：
> The dragon was very scared and lost.
> He wished he had not fought the dragon.
> He wished he had never fought with the dragon.
> He wished he had not fought with the dragon.
```

连续三次"He wished he had ... fought the dragon"。这是 30M 这个尺度模型的典型问题——一旦进入某个高概率 phrase，下一步最高概率往往还是同一个 phrase 的开头。

**怎么缓解**：
- 提高 `temperature` 到 0.9–1.0，把分布拉平。
- 提高 `top_p` 到 0.95，给更多选择。
- 加 `repetition_penalty`（本项目暂未实现）：把已经出现过的 token 的 logits 减一个常数。
- 终极解：训练更大的模型、更多的步数。

---

## 8. 我自己踩过的坑

| 坑 | 症状 | 定位 |
|----|------|------|
| 没做 context cropping | 序列长度超过 block_size 后报错 | `idx[:, -block_size:]` |
| `temperature == 0` 不短路 | 除零 ZeroDivisionError | 在 `if temperature == 0` 分支里走 argmax |
| top-p 没做 "右移一位" | `p=0.5` 时整个分布被剔光 | 移位 + 第 1 个永远保留 |
| top-p mask 没 scatter 回原序 | 似乎能 sample 但内容很怪 | `scatter(-1, sorted_indices, ...)` |
| `model.train()` 没切到 eval | 生成时 dropout 是开的，每次都不一样 | `model.eval()` 在加载 checkpoint 后立即调 |
| 用 `print` 输出生成结果 | 不是 bug，但不符合 "用 logger" 规则——除外：generate.py 是面向用户的程序，**stdout.write 是正确的** | 区分 logging 和 program output |

---

## 9. 复现命令

```bash
# Greedy（确定性）
python generate.py --checkpoint checkpoints/ckpt_step10000.pt \
    --prompt "Once upon a time" --max-new-tokens 200 \
    --temperature 0.0 --seed 42

# Temperature 0.8
python generate.py --checkpoint checkpoints/ckpt_step10000.pt \
    --prompt "Once upon a time" --max-new-tokens 200 \
    --temperature 0.8 --seed 42

# Top-k 40
python generate.py --checkpoint checkpoints/ckpt_step10000.pt \
    --prompt "Once upon a time" --max-new-tokens 200 \
    --temperature 1.0 --top-k 40 --seed 42

# Top-p 0.9（nucleus）
python generate.py --checkpoint checkpoints/ckpt_step10000.pt \
    --prompt "Once upon a time" --max-new-tokens 200 \
    --temperature 1.0 --top-p 0.9 --seed 42

# 组合：top-k 40 + top-p 0.9
python generate.py --checkpoint checkpoints/ckpt_step10000.pt \
    --prompt "Once upon a time" --max-new-tokens 200 \
    --temperature 0.9 --top-k 40 --top-p 0.9 --seed 42
```

---

## 10. 系列总结

回到第一篇开头的设计目标：

- ✅ **不依赖 `transformers`**（除 tokenizer）
- ✅ **手写注意力**
- ✅ **核心模型 < 200 行**（实际 ~190 行）
- ✅ **~30M 参数**（实测 30.0M）
- ✅ **Colab T4 一小时收敛**（实测 83 分钟）
- ✅ **能生成连贯英文短故事**（PPL ≈ 4.7，远低于阈值 15）

整个项目大约 1000 行代码（含训练、数据、测试、推理），跑通完整 GPT 训练 → 推理流程。如果你看到这里——已经把 transformer 这个东西从 LayerNorm 一直到 nucleus sampling 全部走过一遍了。

接下来推荐的练习：

1. **加 KV cache**：现在每次 forward 都从头算，generation 慢。
2. **加 `repetition_penalty` / `no_repeat_ngram_size`**：缓解复读。
3. **换更大的模型**：n_layer=12, n_embd=768 → 124M（GPT-2 small 规格），看 PPL 能压到多低。
4. **换更专业的数据**：从 TinyStories 换到 OpenWebText 或代码数据，看模型行为如何变化。

代码全部在仓库里，欢迎 fork。

→ 回到 [Part 1: 架构篇](./part1_architecture.md) | [Part 2: 训练篇](./part2_training.md)
