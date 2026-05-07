# CLAUDE.md

> Claude Code 在每次对话开始时会自动读取此文件。保持精简（< 200 行），只放真正影响代码生成的内容。

---

## Project Overview

**MiniGPT-from-scratch**：用 PyTorch 从零实现一个 GPT 模型，在 TinyStories 数据集上训练（~30M 参数，~1 小时收敛），能生成连贯的英文故事。

**核心约束：**

- **不依赖 HuggingFace `transformers`**（tokenizer 可用 `tokenizers` 库或 `GPT2Tokenizer`）
- 模型参数量 ~30M（Colab 免费版 T4 可跑；其中 token embedding 约 19M，因 GPT-2 vocab=50257）
- 核心代码 < 1000 行（model.py < 200 行）
- 目标产出：GitHub 开源项目 + Colab Notebook + 技术博客

---

## Tech Stack

- **Language**: Python ≥ 3.9
- **Framework**: PyTorch ≥ 2.0（用 `torch.compile` 加速）
- **Tokenizer**: GPT-2 tokenizer（vocab_size=50257）
- **Dataset**: TinyStories（~3MB 英文短故事）
- **Tracking**: Weights & Biases（首选）或 TensorBoard
- **Training**: 混合精度（FP16 / `torch.cuda.amp`）

---

## Project Structure

```
MiniGPT-from-scratch/
├── CLAUDE.md              # 本文件（项目上下文）
├── README.md              # 面向用户的项目介绍
├── requirements.txt
├── config.py              # 所有超参数（GPTConfig 类）
├── model.py               # 核心：LayerNorm / CausalSelfAttention / MLP / Block / GPT
├── utils.py               # 数据加载 + tokenizer 封装
├── train.py               # 训练循环（含 AMP / 梯度累积 / LR schedule）
├── generate.py            # 推理：贪心 / temperature / top-k / top-p
├── MiniGPT.ipynb          # Colab 一键运行 notebook
├── checkpoints/           # 模型权重（.gitignore）
├── data/                  # TinyStories 原始数据（.gitignore）
└── blog/                  # 技术博客 markdown
    ├── part1_architecture.md
    ├── part2_training.md
    └── part3_generation.md
```

---

## Core Hyperparameters（请勿擅自修改，需先讨论）

定义在 `config.py` 的 `GPTConfig` 类中：

| 参数 | 值 | 说明 |
|------|-----|------|
| `vocab_size` | 50257 | GPT-2 tokenizer 词表 |
| `block_size` | 256 | 上下文长度 |
| `n_layer` | 6 | Transformer 层数 |
| `n_head` | 6 | 注意力头数 |
| `n_embd` | 384 | 嵌入维度（必须能被 n_head 整除） |
| `dropout` | 0.1 | |
| `batch_size` | 64 | |
| `learning_rate` | 3e-4 | AdamW |
| `max_iters` | 10000 | |
| `warmup_iters` | 500 | |
| `grad_clip` | 1.0 | |

预期参数量约 **30M**（含 embedding）。其中：token embedding（与 lm_head 共享权重）约 19.3M，position embedding 约 0.1M，6×Block + final LN 约 10.6M。

---

## Architecture Decisions

- **Position Embedding**：使用**可学习的** position embedding（与 GPT-2 一致），不用 sinusoidal
- **LayerNorm 位置**：使用 **Pre-LN**（norm 在 attention/MLP 之前），比 Post-LN 更稳定
- **Attention Mask**：下三角 causal mask，加在 softmax **之前**（用 `-inf` 填充）
- **Loss**：标准交叉熵，target = input 左移一位（next-token prediction）
- **LR Schedule**：linear warmup（前 500 步）+ cosine decay
- **混合精度**：`torch.cuda.amp.autocast` + `GradScaler`

---

## Common Commands

```bash
# 安装依赖
pip install -r requirements.txt

# 下载并解压数据
bash scripts/download_data.sh

# 训练
python train.py

# 单次推理
python generate.py --prompt "Once upon a time" --max_new_tokens 200 --temperature 0.8

# 跑单元测试
python -m pytest tests/ -v
```

---

## Coding Conventions

- **风格**：PEP8，行宽 ≤ 100
- **注释语言**：英文（方便开源）
- **类型注解**：所有函数签名必须有 type hints
- **导入顺序**：标准库 → 第三方 → 本地
- **Tensor 命名约定**：用 `B / T / C` 表示 batch / time / channel 维度，注释中显式标注 shape
- **日志格式**：`Step {i} | loss {loss:.4f} | lr {lr:.2e} | time {dt:.1f}s`
- **Checkpoint 路径**：`checkpoints/ckpt_step{step}.pt`

---

## Prohibited（禁止做的事）

- ❌ **禁止 import `transformers`**（tokenizer 除外，且仅用 `GPT2Tokenizer`）
- ❌ 禁止用 `nn.MultiheadAttention` —— 必须手写注意力，体现底层理解
- ❌ 禁止用 `nn.TransformerEncoderLayer` / `nn.TransformerDecoderLayer`
- ❌ 禁止 `print` 调试，统一用 logger 或 wandb
- ❌ 禁止跳过单元测试直接写下一个模块
- ❌ 禁止把 checkpoint / wandb 日志 / 数据文件提交到 git

---

## Development Workflow

严格按依赖顺序，每完成一项 git commit 一次：

1. `config.py` → 能 import 不报错
2. `model.py` 的 `LayerNorm` → 单测通过
3. `model.py` 的 `CausalSelfAttention` → 单测通过（含 causal mask 验证）
4. `model.py` 的 `MLP` → 单测通过
5. `model.py` 的 `Block` → 单测通过
6. `model.py` 的 `GPT` → 参数量约 30M
7. `utils.py` 数据加载 → DataLoader 输出 shape 正确（`y` 是 `x` 左移一位）
8. `train.py` 跑通 100 步，loss 稳定下降，无 NaN
9. 接入 wandb，看到 loss 曲线
10. 加入 AMP / 梯度累积 / LR scheduler
11. 完整训练 10000 步，验证集 PPL < 15
12. `generate.py` 贪心解码 → 加 temperature / top-k / top-p

---

## Verification Checklist（修改代码后必须自查）

- [ ] 模型 forward 输出 shape：`(B, T, vocab_size)`
- [ ] 训练 loss 在前 100 步从 ~11 下降
- [ ] 显存占用 < 4GB（FP16 下）
- [ ] 单元测试全部通过：`pytest tests/ -v`
- [ ] 无 `transformers` 库依赖（tokenizer 除外）

---

## References

- 数据集：https://huggingface.co/datasets/roneneldan/TinyStories
- 论文：Attention Is All You Need (Vaswani et al., 2017)
- nanoGPT 参考实现：https://github.com/karpathy/nanoGPT
- Pre-coding checklist：见 `MiniGPT_PreCoding_Checklist.md`

---

## When in doubt

- 模型实现细节有疑问 → 参考 nanoGPT 的对应模块（不要照抄，要自己写）
- 训练不收敛 → 先检查 lr / mask / target 左移，再调 dropout
- 生成乱码 → 先检查 tokenizer 编解码是否对称
- 不确定要不要改超参 → **先问我，不要擅自修改 `config.py`**
