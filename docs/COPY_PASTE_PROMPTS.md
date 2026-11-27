# 直接复制给队友 AI 的提示词

两位队友各复制自己的一整段，粘贴给在**已克隆仓库根目录**工作的 coding agent。当前目标是 12 次训练、48 份测试评价，然后再分析。队友只需把“我是 Person 2”或“我是 Person 3”对应的提示词交给 agent；其中的文件路径和命令都是相对仓库根目录的。仓库不含 CoNLL 原文、处理后数据、模型权重和逐句预测。

## Person 2：直接复制以下整段

> 你是这个三人 NER 项目的 **Person 2 开发 agent**。请直接在当前仓库修改代码、运行测试、记录真实结果；不要只给建议或伪代码。先读 AGENTS.md、docs/TEAM_START_HERE.md、docs/DATA_FORMAT.md、docs/TRAIN_TEST_GUIDE.md、docs/PERSON_2_GUIDE.md，以及 src/models/base.py、src/models/registry.py、scripts/train_model.py、scripts/test_model.py。当前只做首批 12 次实验：CRF 的 10% 和 100% 已由 Person 1 完成；**你的当前目标是完成 lr_raw 的 10% 和 100% 两次训练及各自四种测试条件，共 2 次训练、8 份评价**。PCA64/128、CRF 消融和多 seed 暂缓。
>
> 先在 feature/crf-lr 分支工作。检查 data/raw/train.txt、valid.txt、test.txt 是否存在；如果没有，先完成不依赖真实数据的代码与测试，再向我明确索取学校允许使用的文件。不要把原文、处理后 JSONL、checkpoint、逐句预测或错误案例提交到公开 GitHub。数据就绪后运行 python -m scripts.prepare_data --source local --train data/raw/train.txt --validation data/raw/valid.txt --test data/raw/test.txt --seed 42，核对划分句数和真实 doc_id。设备方面，lr_raw 可以用 CPU（建议 16 GB 内存）运行；先用 10% 测耗时，有学校 T4 时可用 GPU 加速冻结特征抽取。设备详情见 docs/TEAM_START_HERE.md。
>
> 在 src/models/lr_pca/model.py 实现 **lr_raw** 的 train、predict、save、load。调用 Hugging Face Transformers 提供的通用 distilbert/distilbert-base-cased，冻结 encoder，只抽取原始词的首个 subword 表示；调用 scikit-learn LogisticRegression 训练词级标签分类器。长句要处理完整、每个原词恰有一次预测。按训练子集的真实 doc_id 做三折 GroupKFold 选 LR 参数，任何拟合步骤都不能看官方 Validation/Test；不可使用已在 CoNLL 上微调的 NER checkpoint，也不能用 POS/chunk。预测返回与输入 tokens 同长的 IOB2 标签，统一使用 src/data/dataset.py 的标签映射和修复规则。保存后重新加载应给出同样预测。尽量只改你负责的模型目录和自己的测试；共享接口若确需更改，先说明给 Person 1。
>
> 先运行 python -m unittest discover -s tests -v，再补能检验词级对齐、文档分组无泄漏、save/load 一致性和预测长度的测试。实现完成后依次运行：python -m scripts.train_model --model-id lr_raw --fraction 0.1 --seed 42；python -m scripts.test_model --run-id lr_raw_10pct_seed42；将 fraction 改为 1.0 后训练，再测试 lr_raw_100pct_seed42。训练脚本会检查共享 manifest、提供 validation_variants，测试脚本会生成四条件预测、metadata、evaluation.json/csv 和错误案例。若依赖下载或计算资源受限，继续完成代码与 CPU 小样例测试，只把真实运行标为未完成，并准确报告阻塞原因。
>
> 最后给我：改动文件、测试通过情况、两次训练各自的 checkpoint 和四条件实体级 F1/耗时、尚未完成的部分，以及可供 Person 1 复核的产物路径。只提交代码/配置/不含原文的摘要到分支，准备 PR；不得虚构训练结果。

## Person 3：直接复制以下整段

> 你是这个三人 NER 项目的 **Person 3 开发 agent**。请直接在当前仓库修改代码、运行测试、记录真实结果；不要只给建议或伪代码。先读 AGENTS.md、docs/TEAM_START_HERE.md、docs/DATA_FORMAT.md、docs/TRAIN_TEST_GUIDE.md、docs/PERSON_3_GUIDE.md，以及 src/models/base.py、src/models/registry.py、scripts/train_model.py、scripts/test_model.py。当前只做首批 12 次实验：CRF 的 2 次已完成；**你的目标是四种 DistilBERT 方法各在 10% 和 100% 训练量下运行一次，seed=42，共 8 次训练、32 份四条件测试评价**。模型 ID 为 distilbert_clean、distilbert_aug、distilbert_consistency、distilbert_balanced。额外 seed、PCA 和 CRF 消融暂缓。
>
> 先在 feature/distilbert 分支工作。检查 data/raw 中学校允许使用的 CoNLL train.txt、valid.txt、test.txt；若没有，先实现代码与不依赖真实数据的测试，再向我索取合法数据。不要将语料、处理后 JSONL、checkpoint、逐句预测或带原文的错误案例提交到公开 GitHub。数据准备统一运行 python -m scripts.prepare_data --source local --train data/raw/train.txt --validation data/raw/valid.txt --test data/raw/test.txt --seed 42；不得自行重新抽取 10% 子集或生成另一版测试噪声。
>
> 在 src/models/distilbert/model.py 实现 train、predict、save、load，调用 PyTorch 与 Hugging Face Transformers 的 **通用** distilbert/distilbert-base-cased 和 token classification head；不要使用已在 CoNLL NER 微调的模型。四种方法分别是：Clean-only 用原文 CE；Aug 用原文与 config["noisy_train_rows"] 的 CE 平均；Ordinary 在 Aug 上加所有有效原词的平均 JSD；Balanced 在 Aug 上加 gold 实体词和 gold O 词各占 50% 的 JSD，若 batch 仅有一组则该组占 100%。lambda 从 0.1、0.5、1.0 用验证集选。clean/noisy 可能有不同 subword 切分，必须按原始 word index 对齐，只让每词首个 subword 参与损失；忽略 special tokens 和 padding，数值计算稳定。长句用重叠窗口且最终每词只输出一次。详细公式和测试样例见 docs/PERSON_3_GUIDE.md。
>
> 保持四种方法相同的通用预训练起点、原句 batch、优化器更新预算和 checkpoint 选择规则。只依据 config["validation_variants"] 中 Clean、Lowercase、Typo-10 的**平均严格实体级 F1**选择 checkpoint/参数，绝不能用 Test 选择。实现后先跑 python -m unittest discover -s tests -v，并补充 JSD 稳定性、全 O/全实体 batch、word 对齐、长句覆盖、save/load 和预测长度测试。先跑 10% 的四种方法，再跑 100%。每个 model_id 依次运行 python -m scripts.train_model --model-id MODEL_ID --fraction 0.1 --seed 42 和 python -m scripts.test_model --run-id MODEL_ID_10pct_seed42；100% 时改 --fraction 1.0、run_id 中改为 100pct。
>
> 首选本机 RTX 3060 Laptop 训练。先运行 nvidia-smi 查看实际显存，并确认 torch.cuda.is_available()。若显存约 6 GB，从 10% 数据、micro-batch=1 试跑，用梯度累积维持有效 batch；显存不足尝试混合精度、缩短窗口或梯度检查点。一致性方法可能比 Clean-only 更占显存。若 batch=1 仍无法稳定运行，再把学校 T4（16 GB 显存）作为备用，不要假设云资源已分配。若任何 GPU 暂不可用，先把模型代码、CPU 小样例测试、配置和可执行命令全部完成，再准确报告阻塞；不要编造结果。每次真实运行记录 checkpoint、四条件预测、metadata、训练时间、推理时间和实际硬件。最后给我：改动文件、测试情况、8 次训练与 32 份评价的真实完成情况、四条件实体级 F1/耗时、产物路径和阻塞项。只提交代码/配置/不含原文的摘要到分支，准备 PR。

