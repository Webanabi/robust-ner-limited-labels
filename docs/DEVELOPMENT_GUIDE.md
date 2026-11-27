# 小组开发总指南

## 1. 项目是什么

本项目做英文平面命名实体识别（NER）：给每个原始词一个 IOB2 标签，最终找出 PER、ORG、LOC、MISC 实体的边界和类别。研究问题是：只用 10% 标签时模型退化多少；大小写和拼写噪声下谁更稳；实体平衡一致性损失是否提高噪声文本的严格实体级 F1。开题报告提出的收益是待验证假设。原报告的分工已被本指南第 3 节覆盖，其他实验约束仍保留。

CRF 提供可解释、CPU 可运行的传统基线；冻结 DistilBERT 表示加 LR 检验预训练语义特征而不做端到端微调；四种 DistilBERT 方法从同一通用预训练权重起步，逐步隔离噪声增强、普通一致性和实体平衡的贡献。它们共同使用同一数据、标签顺序、噪声文本和评估脚本。不能使用已在 CoNLL-2003 上微调的 NER checkpoint。

## 2. 全流程

合法获取原始数据 → 验证官方划分与文档边界 → JSONL/统计 → 同 seed 的 10% 句子 ID 和固定验证/测试噪声 → 各成员训练与选择参数 → 原始单词级预测 JSONL + metadata → 公共严格实体级评估 → 聚合、作图、错误分析 → Streamlit 推理演示 → 报告与演示文稿。

    合法 CoNLL 源文件
            ↓
    解析 + 划分/标签/文档审计
            ↓
    公共 JSONL ──→ 同 seed 采样 + 固定噪声
            ↓                    ↓
       CRF / LR/PCA       四种 DistilBERT 方法
            └──────────┬──────────┘
                 原词级预测 + metadata
                         ↓
                 严格实体级评估
                         ↓
               汇总 / 图表 / 错误分析
                         ↓
                  报告与本地演示

开题报告给的句数为 Train 14041、Validation 3250、Test 3453。prepare_data 会记录实际句数与差异，不会替换/截断以凑数。只有保留 -DOCSTART- 的本地文件能为 Person 2 提供真实文档边界。HF 可作为获取句子和标签的备选，但当前接口将 doc_id 标为 UNAVAILABLE，因此不得以该数据运行“按原文档分组 CV”。无论来源，须检查数据许可，不把受限原始文本或由其生成的可还原数据推送到公开仓库。

## 3. 三人分工与边界

Person 1 负责实验协议、公共数据/扰动/评估/可视化、界面、集成和复现检查，维护 configs、src/data、src/evaluation、src/analysis、scripts、app.py。Person 2 负责 CRF、大小写与词缀消融、冻结 DistilBERT + LR 和 PCA 64/128，修改 src/models/crf 与 src/models/lr_pca。Person 3 负责 Clean、Aug、Ordinary Consistency、Balanced Consistency 四种微调，修改 src/models/distilbert。Person 2 可在自己的任务完成后帮 Person 3 跑实验。三人共同写最终报告和 presentation。

共享接口、label map、评价规则由 Person 1 管理。其他人遇到协议不足应提出具体修改并让 Person 1 同步，避免分支各自悄悄改变定义。Person 2 和 Person 3 的专用步骤见各自指南。

## 4. 目录和文件

- README.md：快速开始；AGENTS.md：协作规则；EXPERIMENT_STATUS.md：进度概览；EXPERIMENT_MATRIX.csv：逐运行状态、路径和参数。
- configs/default.yaml：种子、扰动、checkpoint 与 lambda；configs/experiments.yaml：核心/补充/可选矩阵；configs/label_map.json：标签 ID。
- src/data/dataset.py：验证 JSONL、BIO 修复、实体 span；preprocess.py：本地 CoNLL/HF 读取；sampling.py：共享子集；noise.py：扰动；statistics.py：分布。
- src/models/base.py：train/predict/save/load 合同；registry.py：模型 ID；三个模型目录：Person 2/3 的待实现骨架与 README。
- src/evaluation/metrics.py：严格实体级指标；error_analysis.py：互斥的一对一错误匹配；evaluate.py：文件校验和 CSV/JSON 导出。
- src/analysis/aggregate.py：只收真实评估结果、计算多 seed 平均/标准差和 Clean drop；visualization.py：由真实 CSV 画图。
- scripts/prepare_data.py、train_model.py、test_model.py、predict.py、run_evaluation.py、generate_report_assets.py、init_experiment_matrix.py：命令入口；app.py：模型加载后的交互推理；src/demo.py：可单独测试的字符偏移与高亮逻辑。
- data、checkpoints、results：本地/云端生成内容，Git 忽略；tests：不依赖 GPU 的公共测试。

## 5. 环境与数据

建议 Python 3.10+。公共 CPU 环境安装 requirements.txt；训练环境另装 requirements-gpu.txt。安装过程见 README。真实 CoNLL 数据需由有权限的人依据课程和原始数据许可取得。把含 -DOCSTART- 的 train、valid、test 文件放在 data/raw，并运行 README 中的 prepare_data 命令。已处理的原始词数据同样不要公开提交。初次准备后检查 source_audit.json、各 statistics.json、10% manifest 和 noise_audit；手工抽查数句。测试数据和验证数据保持完整。

若本地原始文件无法取得，--source hf 是可选下载路径，需有网络和 datasets 库；无法确认文档边界时，Person 2 先不要声称完成按文档交叉验证。当前公共模块没有访问真实数据或执行 GPU 训练。

## 6. 协议和运行

输入/预测格式与字段示例见 DATA_FORMAT.md。各模型必须实现 NERModel.train、predict、save、load，预测长度等于输入原始 tokens 数。用 predict_checked 检查标签。公共 scripts/train_model.py 读取并核对训练数据，再调用模型的 train/save；scripts/test_model.py 在四种固定条件下调用 load/predict、评估并更新进度。CRF 已可真实训练和推理；LR/PCA、DistilBERT 仍是 NotImplementedError 骨架。命令和产物见 TRAIN_TEST_GUIDE.md。每个测试运行会保存 metadata.json，字段详见 DATA_FORMAT.md。

tests/test_public.py 中的 FixtureModel 是最小可运行接口示例，仅用于验证 export_predictions；它总预测 O，不能作为真实 NER 结果。队友应在各自骨架上实现真实 train/predict/save/load，再用公共预测脚本导出。

评估命令见 README。传入的 gold 是对应条件的公共 test_clean/lowercase/typo10/typo20 JSONL；预测中的 token 必须完全一致，评估器按 ID 对齐，不信任模型自己附带的 gold。缺句、多句、重 ID、长度错误、非法标签均立即报错。预测里的非法 I-X 在 span 提取前统一改为 B-X。严格实体匹配要求起点、终点、类型全相同。

错误匹配顺序：完全正确先移除；同边界异类型记 wrong_type；剩余有交集者按重叠词数降序一对一配对记 boundary；剩余 gold 记 missed，剩余预测记 false。每个 span 最多参与一个错误对。报告类别数与示例时说明这一规则。

从多个 evaluation.json 生成 runs.csv、summary.csv、drops.csv 与图，命令见 README。标准差只在至少两个 seed 时给值，单 seed 不展示为“0 波动”。可视化模块不补不存在的结果。Person 2 的 PCA 二维可视化需要另外导出特征与标签；没提供就不生成。

## 7. 实验矩阵与选择

核心先跑四种 DistilBERT × 100%/10% × seed42；每个模型都在四种测试条件评估。CRF 和 LR 原始特征为补充；CRF 两消融、PCA64/128 和另外两个 seed 为可选扩展。时间不足应如实列出未运行项。configs/experiments.yaml 是计划，不表示已运行。状态以 EXPERIMENT_STATUS.md 和真实结果文件为准。

Person 2：仅在当前训练子集内进行三折 GroupKFold；PCA 和其他拟合变换每折重拟合。Person 3：以 validation Clean、Lowercase、Typo-10 的实体 F1 均值选 checkpoint、lambda∈{0.1,0.5,1.0}。Test 不用于调参。神经实验共享原始句 batch、初始化来源和近似相同优化器更新步数；增强方法每步的额外计算量单独报告。对比推理速度时固定硬件、batch size、计时口径。

## 8. 噪声和可复现性

训练每个原句产生一个扰动副本；lower 和 typo 等概率选一。lower 从有大小写变化的词中独立按 30% 抽取；typo 从至少 4 个 ASCII 字母、内部有不同字母的词中独立按 10% 抽取，交换两内部不同字母，首尾不变。测试 Lowercase 全词小写；Typo-10/20 分别按 10%/20% 选合格词。所有扰动固定于 seed+句子 ID+条件，不随遍历顺序改变。实际 changed/eligible 由 audit 文件计算并在报告注明；Bernoulli 抽取的实际比例不保证刚好等于名义比例。不同验证/测试 seed 避免把同一随机副本用于选择和最终汇报。

## 9. 云端与 GitHub

云端操作见 GOOGLE_CLOUD_GUIDE.md，包括学校发的 Notebook/T4 模板步骤和授权待确认项。不要假设 GPU 已分配。main 保存公共稳定代码；feature/crf-lr 给 Person 2，feature/distilbert 给 Person 3，feature/evaluation 给 Person 1 的评估工作。先拉取 main，再建分支；小步提交并附运行命令/测试结果；通过 PR 合并。需要共享 API 更改时先提出，避免三个分支分别改标签顺序或 JSONL 字段。模型权重、原始数据、缓存和大结果不进 Git；可用受控渠道传预测和配置。

## 10. 验收与常见问题

最低验收：CPU 测试通过；真实数据来源、大小、许可与文档边界记录；同 seed 共享 10% ID；固定扰动与标签不移位；真实模型导出原词级预测；评估严格匹配且覆盖所有句；真实 metadata/配置/运行日志可追溯；图表仅来自真实结果；报告明确完成和未完成项目。测试样例保存在 tests 中，不是实验成绩。

报错排查：doc_id=UNAVAILABLE 时换本地原始文件；预测覆盖错误时核对 split/condition 和 ID；token 不匹配时用公共噪声文件而不是模型自行扰动；BIO 非法 gold 应核对解析和源文件，预测非法 I-X 由共享规则修复；CUDA 内存不足先降 micro-batch、用梯度累积并记录变更；没有训练好的 manifest 时 Streamlit 只显示提示；网络下载受限时用本地合法数据，不报告未实际执行的结果。

### 与开题报告的差异

原报告按成员名字分配 CRF、噪声和 demo；本项目按用户新分工改为 Person 1 公共模块、Person 2 两类基线、Person 3 四种微调。原报告的云端 16 GB GPU 是假设，学校提供的 T4 模板也需授权和配额核实。当前工作区已在本地准备数据，但 Git 忽略原始和处理后文本；尚无真实实验成绩，计划中的模型方法均待队友实现。

