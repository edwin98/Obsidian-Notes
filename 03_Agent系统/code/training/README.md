# 新版训练代码怎样学与运行

先理解正文08，再按本页运行。默认日常Agent不会启动训练；训练数据必须先审核并按原问题分组。

## 1. 不加载模型的数学演练

~~~powershell
uv sync --extra training
uv run python -X utf8 -m training.inspect_batch
~~~

objectives.py 展示 labels、因果位移、候选序列概率与固定参考；预期可看到约0.4581的SFT损失与0.4032的DPO损失。这是数学例子，不是任务成绩。

## 2. 不下载权重的完整结构自检

~~~powershell
uv run python -X utf8 -m training.tiny_smoke
~~~

它创建极小随机Qwen结构和本地tokenizer，构造真实批次，CPU分别做一步SFT与DPO，再检查保存与合并。默认临时文件自动清理；--output-dir 指定一个新目录可保留查看。

toy训练/验证材料只检查结构，不属于质量评测。不能把这一步的loss或偏好准确率当实际调查能力。

## 3. 从专家审核导出数据

TaskService.prepare_step 冻结输入；实验记录保留 context_snapshot_id。learning.export_decision_reviews 接收明确的 DecisionReview，目标是与部署一致的 PlanDecision。

新版SFT包含messages，DPO保留prompt角色列表及完整chosen/rejected。元数据包含source_group、reviewer、target_schema与快照引用。没有确认的记录不会导出。

## 4. 统一分组，再训练

~~~powershell
python -X utf8 -m training.prepare_investigation_data --input-dir reviewed --output-dir prepared
~~~

输出SFT/DPO的train、validation、test及groups.json。同一原问题、改写和修复跟进必须同组。哈希按80/10/10分配，独立问题少时某个集合可能为空；不能拿最终测试数据凑验证集。

旧ExperimentPlan目标会规范成PlanDecision。训练入口要求显式独立验证集，不再次随机拆分。

## 5. 检查真实SFT批次

以下命令需要已准备的数据；模型未缓存时会按所选模型加载权重。先用小模型和CPU学习，再按资源决定设备配置。

~~~powershell
python -X utf8 -m training.train_sft --model_name_or_path Qwen/Qwen3-0.6B --data_path prepared/sft_train.jsonl --validation_path prepared/sft_validation.jsonl --output_dir checkpoints/sft-inspect --cpu --inspect-only
~~~

检查器打印整理器生成的实际张量形状、目标数量和首个目标解码。应主要看到助手计划，而不是用户消息或工具测量。模板不支持助手掩码时修模板，不静默退化成全文损失。

## 6. SFT训练与DPO跟进

~~~powershell
python -X utf8 -m training.train_sft --model_name_or_path Qwen/Qwen3-0.6B --data_path prepared/sft_train.jsonl --validation_path prepared/sft_validation.jsonl --output_dir checkpoints/sft --cpu --merge
python -X utf8 -m training.train_dpo --sft_model_path checkpoints/sft/merged --data_path prepared/dpo_train.jsonl --validation_path prepared/dpo_validation.jsonl --output_dir checkpoints/dpo --cpu
~~~

参数只是教学起点。硬件支持时可去掉--cpu并明确选择--bf16；分布式配置按实际环境提供。DPO显式加载独立冻结SFT参考，内存占用更高，但便于看清reference与policy。

SFT足够就不做DPO。偏好对不足、候选同样合理或来源不明确时，先补数据，不自动把模型评审当专家真值。

## 7. 输出分别代表什么

- adapter/：适配器与tokenizer；需要对应基座。
- merged/：只有显式--merge且单进程时输出完整模型。
- checkpoint-*：训练器周期保存的恢复点，可含优化器等状态。
- trainer_state.json：状态元数据，单独它并不等于完整优化器恢复点。
- training_manifest.json：参数与输入文件哈希，帮助追溯。

Windows请使用-X utf8，避免第三方模板默认GBK读取失败。真实训练中先固定依赖版本、核对目标位置并短试训，再通过完整Agent任务决定是否采用模型。

## 8. 历史脚本与其他框架

generate_sft_data.py、clean_data.py、prepare_dpo_data.py保留历史结论型数据参考；它们不直接承担新实验计划清洗。training_llamafactory 是可选转换路径，先分组再分别转换，保留来源元数据，不静默删掉工具消息。

完整模块与调用关系见 [学习指南](../学习指南.md)。
