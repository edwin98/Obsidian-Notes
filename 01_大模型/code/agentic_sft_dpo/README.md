# Agentic SFT + DPO 实践（单卡 LoRA）

用一张 12–16GB 的显卡，从开源数据出发，把 **Qwen2.5-1.5B-Instruct** 训练成一个会边想边调用工具的数学 Agent：先用 **SFT** 让它学会多轮工具调用，再用 **DPO** 让它在自己真实会犯的错上变好。每一步都在真实执行工具的 Agent 循环里评测，能直接看到行为和指标的变化。

对应笔记：[[01_大模型/02_训练与对齐/01_SFT与参数高效微调]]、[[01_大模型/02_训练与对齐/02_偏好对齐算法_PPO_DPO_GRPO]]、[[03_Agent系统/08_Agent模型后训练SFT_DPO]]。`03_Agent系统/code/training` 讲的是怎样从专家审核记录导出数据；这里换成公开数据，从头到尾真正训一遍、测一遍。

## 一、任务设计：为什么选“GSM8K + 计算器”

| 需要的东西 | 本项目里是什么 |
| --- | --- |
| 开源的 Agent 轨迹数据 | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) 参考解答里自带 `<<48/2=24>>` 计算器标注，每个标注都改写成一次 `calculator` 工具调用，于是 7473 道题就变成了 7473 条**多轮工具调用轨迹** |
| 可以真实执行的环境 | `common.py` 里的安全计算器（只解析算术 AST）。评测和采样时，模型发出的每个调用都被真实执行并回填 |
| 自动判对错的奖励 | 标准答案。正确就是 1，错误就是 0，不需要额外的打分模型 |
| 能直接看出来的效果 | 原模型通常不调工具、长篇心算；SFT 后会按格式调工具；DPO 后，同样的格式下答对率和稳定性应该进一步提高 |

一条 SFT 轨迹长这样（`prepare_data.py` 运行时会打印）：

```text
[user]      Stefan ... tip 20% of the total ... what is the total amount?
[assistant] 🔧 calculator("4*20")
[tool]      80
[assistant] The total cost of the entrees is 4 * $20 = $80.  🔧 calculator("80+10")
[tool]      90
...
[assistant] The total cost with tip is $90 + $18 = $108
            Final answer: 108
```

## 二、环境准备

在本目录打开 PowerShell：

```powershell
cd '01_大模型/code/agentic_sft_dpo'
uv sync                       # 从 PyTorch 官方源装 CUDA 12.8 版 torch，第一次约 3GB
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

必须打印 `True` 和显卡名。`pyproject.toml` 已把 torch 指向 cu128 源，30/40/50 系显卡都能用。如果显示 False，先运行 `nvidia-smi`，确认驱动支持 CUDA 12.8；驱动较老时，把 `cu128` 改成 `cu126` 再 `uv sync`。

模型（约 3GB）和数据集（几 MB）第一次运行时自动从 Hugging Face 下载，缓存在 `~/.cache/huggingface`。想换模型可以设环境变量，例如 `$env:BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"`（更快，效果差异更明显）。

## 三、运行

### 一键跑完

```powershell
.\run_all.ps1 -Quick     # 快速版：600 条 SFT、150 道 DPO 题、100 道评测题
.\run_all.ps1            # 完整版：2000 / 400 / 300
```

用时取决于显卡，以下只是粗估：快速版大约半小时到一小时，完整版大约 1.5–3 小时。大部分时间花在 Agent 的**生成**上，包括评测和 DPO 采样；训练本身相对较快。建议先跑 Quick，确认能跑通、看到趋势后再跑完整版。

### 分步运行（推荐第一次这样做，每步停下来看输出）

```powershell
uv run python -X utf8 prepare_data.py                          # 1. 开源数据 → 多轮轨迹
uv run python -X utf8 evaluate.py --name base                  # 2. 原模型当 Agent 的表现（基线）
uv run python -X utf8 train_sft.py --inspect                   # 3a. 看一个 batch：哪些 token 参与损失
uv run python -X utf8 train_sft.py                             # 3b. LoRA SFT → outputs/sft
uv run python -X utf8 evaluate.py --name sft --adapters outputs/sft
uv run python -X utf8 build_dpo_pairs.py --adapters outputs/sft  # 4. SFT 模型在新题上采样，环境判对错 → 偏好对
uv run python -X utf8 train_dpo.py                             # 5. 轨迹级 DPO → outputs/dpo
uv run python -X utf8 evaluate.py --name dpo --adapters outputs/sft outputs/dpo
uv run python -X utf8 evaluate.py --compare base sft dpo       # 6. 三个阶段对比表
uv run python -X utf8 demo.py --index 3                        # 7. 同一道题，看三个模型的逐轮轨迹
```

## 四、每一步该看什么

| 步骤 | 重点看 | 说明 |
| --- | --- | --- |
| prepare_data | 示例轨迹、平均工具调用次数（约 3 次） | 想一想：为什么第一轮 assistant 的内容可以为空，只发工具调用？ |
| evaluate base | `used_tool_rate`、`final_format_rate`，再翻 `outputs/eval_base_trajs.jsonl` | 原模型能不能按协议调工具？失败的是格式、算术还是推理？ |
| train_sft --inspect | `=== LEARN ===` 的片段 | 只有 assistant 写的思考、`<tool_call>` 和 `<|im_end|>` 参与学习；system、题目、**工具返回**都不学 |
| train_sft | 验证损失下降 | 损失下降只说明“更会模仿”，Agent 是否真的变好要看下一步评测 |
| evaluate sft | 工具使用率、格式率应接近 100%，正确率看变化 | 正确率**不一定**涨：1.5B-Instruct 原本就会长链推理，GSM8K 的参考解答较短，模仿短解答可能损失一些推理。这正是需要 DPO 的原因 |
| build_dpo_pairs | “全对 / 全错 / 可配对”的题目数，rejected 的错误类型分布 | 全对或全错的题没有对比信号。可配对的题太少时，提高 `--k` 或 `--temperature` |
| train_dpo | `reward_acc` 上升到明显高于 0.5，`margin` 变大；`chosen_logp` 别掉太多 | 两边概率一起下降是 DPO 的常见现象。下降太多时，调小 `--lr`，或加 `--sft_alpha 0.2` |
| compare | 三列指标 | 输出的“标准误”提醒你：300 题时，正确率相差 2–3 个百分点可能只是噪声 |

## 五、文件与阅读顺序

| 文件 | 内容 | 建议断点 |
| --- | --- | --- |
| [common.py](common.py) | 系统提示、工具 schema、计算器、输出解析、**多轮对话的损失掩码**、模型加载 | `encode_conversation` |
| [prepare_data.py](prepare_data.py) | GSM8K 计算器标注 → tool-calling 轨迹 | `solution_to_messages` |
| [agent.py](agent.py) | 批量 Agent 循环：生成 → 解析 → 执行 → 回填 | `_step` |
| [evaluate.py](evaluate.py) | 评测指标与对比表 | `summarize`（在 agent.py） |
| [train_sft.py](train_sft.py) | 手写 LoRA SFT 循环（没有用 Trainer，便于看清每一步） | `sft_loss` |
| [build_dpo_pairs.py](build_dpo_pairs.py) | on-policy 采样 + 环境奖励 → 偏好对 | 配对的循环 |
| [train_dpo.py](train_dpo.py) | 多轮轨迹级 DPO；参考模型通过关闭 LoRA 得到 | `dpo_step`、`sequence_logps` |
| [demo.py](demo.py) | 同一道题，三个模型的轨迹并排打印 | — |

## 六、关键设计（也是面试常问点）

1. **多轮 SFT 只在 assistant 段计算损失。** 工具返回由环境产生，模型不需要学会“预测计算器输出”。如果把工具返回也算进损失，模型会倾向于自己编工具结果。掩码的做法是渲染两次前缀、比较字符区间，再映射到 token。这依赖 chat template 的“前缀一致性”，代码里用 `assert` 检查。换成 Qwen3 这类会改写历史轮次的模板时，这个断言会失败，需要换一种方式计算掩码。
2. **结尾的 `<|im_end|>` 也要学。** 不学的话，模型不知道什么时候该停下，把控制权交回给运行时。
3. **DPO 数据为什么自己采样，而不是下载公开偏好集？** DPO 压低的是 rejected 的概率。用当前模型自己会犯的错做 rejected（on-policy），信号最强。公开偏好数据里的错误，当前模型往往根本不会犯。这里的“偏好”来自可验证奖励：答案对或错，和 GRPO/RLVR 的思路相通，DPO 可以看作它的离线、成对版本。
4. **轨迹级 DPO，掩码与 SFT 一致。** chosen 和 rejected 是整段多轮轨迹，log π 只累加 assistant token。两条轨迹的工具返回不同，但这些 token 不算进 log π，所以不会污染比较。代价是信用分配比较粗：整条轨迹一起被抬高或压低，看不出是哪一步错了。进阶做法是 step-level DPO，只比较第一处分叉的那一轮。
5. **参考模型不占第二份显存。** SFT LoRA 先合并进权重，再挂一层新的 DPO LoRA；`model.disable_adapter()` 就是 π_ref。
6. **长度偏差。** 序列 log π 是 token 求和，较长的 rejected（例如回合用完还没作答）天然 log π 更低。观察 `rejected_reason` 的分布，想想这会不会让模型学到“写短一点”而不是“算对”。

## 七、显存不够或效果不对

| 现象 | 处理 |
| --- | --- |
| SFT 显存不足（OOM） | `--micro_batch 2 --grad_accum 8`，或 `--max_len 768` |
| DPO 显存不足 | `--micro_batch 1 --grad_accum 8`。每个偏好对要做 4 次前向：策略和参考，各算 chosen 与 rejected |
| 评测或采样显存不足 | `--batch_size 16` |
| SFT 后正确率下降 | 正常现象，可以讨论：调小 lr（`1e-4`）、减少步数，或者混入一部分原模型自己答对的轨迹（拒绝采样微调，RFT） |
| 可配对的题太少 | `--k 8 --temperature 1.1` |
| DPO 的 reward_acc 卡在 0.5 左右 | 把 lr 调到 `5e-5`，或者 `--epochs 3`；检查 `dpo_pairs.jsonl` 里的 chosen 和 rejected 是否真的不同 |
| DPO 后变差 | 看 `chosen_logp` 是否大幅下降。可以加 `--sft_alpha 0.2`、调小 lr，或把 β 调大到 0.3 |

## 八、冒烟记录（不是效果结论）

2026-10-31 在一台无 GPU 的 2 核机器上，用 Qwen2.5-0.5B-Instruct 跑通了完整流程：120 条 SFT 数据训练 15 步，8 道题各采样 3 条，只得到 2 个偏好对，评测只用 6 道题。

| 指标 | base | sft | dpo |
| --- | --- | --- | --- |
| 使用工具的比例 | 16.7% | 100% | 100% |
| 按 Final answer 格式收尾 | 33.3% | 100% | 100% |
| 正确率（n=6，噪声极大） | 0% | 16.7% | 33.3% |

这组数据只能说明两点：代码链路是通的；**很少量的 SFT 就足以改变 Agent 的行为格式**。正确率和 DPO 的效果要在你的 GPU 上按默认规模跑出来再判断。真实结果出来后，可以替换这一节。

## 九、延伸练习

1. 把 `train_sft.py` 改成 TRL 的 `SFTTrainer`，对比 `assistant_only_loss` 需要模板里有 `{% generation %}` 标记，以及 Qwen2.5 模板为什么没有这个标记。
2. 实现 step-level DPO：找到 chosen 和 rejected 第一处不同的 assistant 轮，prompt 取共同前缀，只比较这一轮。
3. 把 `build_dpo_pairs.py` 的输出改成 RFT 数据：只保留答对的轨迹，再做一次 SFT。对比 RFT 和 DPO 的效果。
4. 换一个公开的通用函数调用数据集做 SFT，例如 `NousResearch/hermes-function-calling-v1`，检查数学 Agent 能力是否被冲淡（灾难性遗忘）。
5. 用 `$env:BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"` 重跑一遍，比较不同大小的模型从 SFT 和 DPO 中获益多少。
