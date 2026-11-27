# 队友从这里开始

## 当前目标

先完成 **12 个训练运行**：6 种模型配置各用 10% 和 100% 训练数据、统一 seed=42。每个训练好的模型都要在 Clean、Lowercase、Typo-10、Typo-20 四种固定测试文本上评估。因此验收时应有 **12 个真实 checkpoint、48 份测试评价**，然后再由三人一起分析结果、写报告和 presentation。其他 seed、PCA、CRF 特征消融都暂缓；它们出现在完整计划中，不属于这一阶段。

| 负责人 | 模型配置 | 训练量 | 当前状态 |
|---|---|---|---|
| Person 1 已起跑，Person 2 复核 | crf | 10%、100% | 两次训练及八项测试已完成 |
| Person 2 | lr_raw（冻结 DistilBERT + LR） | 10%、100% | 模型代码待实现 |
| Person 3 | distilbert_clean、distilbert_aug、distilbert_consistency、distilbert_balanced | 每种 10%、100% | 八次模型代码与训练待完成 |

当前首批进度：2/12 个训练运行、8/48 份测试评价。以 EXPERIMENT_MATRIX.csv 的对应行及真实结果文件核对，绝不把 planned 改成 completed 来凑数。

## 克隆仓库后先做什么

1. 先读根目录 AGENTS.md、README.md、docs/DATA_FORMAT.md、docs/TRAIN_TEST_GUIDE.md，再读自己的 PERSON_2_GUIDE.md 或 PERSON_3_GUIDE.md。AI 助手也应按这个顺序读。
2. 运行 python -m unittest discover -s tests -v，确认公共模块正常。Person 2 在 CPU 上可安装 requirements-crf.txt；训练冻结编码器和神经模型需要 requirements-gpu.txt 及实际计算资源。
3. GitHub 仓库**不包含** CoNLL 原文、处理后 JSONL、checkpoint、逐句预测和错误案例。每人应按课程允许的来源获取保留 -DOCSTART- 的 train.txt、valid.txt、test.txt，放到 data/raw，然后运行：

       python -m scripts.prepare_data --source local --train data/raw/train.txt --validation data/raw/valid.txt --test data/raw/test.txt --seed 42

   检查 data/processed/source_audit.json 中的三个句数为 14041、3250、3453，document_boundaries_available=true。Person 1 本地下载文件的 SHA-256 记在 docs/DATA_PROVENANCE.md，可用来确认是否同源。若学校另有受控分发渠道，优先使用该渠道；不要将语料传到公开 GitHub。
4. 建自己的分支：Person 2 用 feature/crf-lr；Person 3 用 feature/distilbert。公共接口变更先告诉 Person 1。小步提交，开 PR 到 main；PR 写明运行命令、测试结果和未完成项目。

## 给 Person 2 及其 AI 的任务

Person 2 负责 src/models/lr_pca/model.py 中的 **lr_raw**。使用通用 distilbert/distilbert-base-cased 抽取冻结词级表示，再用 scikit-learn LogisticRegression 分类；只取原词的首个 subword，不使用已在 CoNLL 上微调的 NER checkpoint。训练子集内按真实 doc_id 做三折 CV；不得用 Test 挑参数。实现 NERModel 的 train/predict/save/load，补必要测试。PCA64/128 属于下一阶段，先不要抢占两次 lr_raw 训练。

完成接口后依次运行：

    python -m scripts.train_model --model-id lr_raw --fraction 0.1 --seed 42
    python -m scripts.test_model --run-id lr_raw_10pct_seed42
    python -m scripts.train_model --model-id lr_raw --fraction 1.0 --seed 42
    python -m scripts.test_model --run-id lr_raw_100pct_seed42

交付：代码、测试、2 个 checkpoint、8 份测试评价及各运行 metadata。让 Person 1 核对预测文件中每句 word 级 BIO 标签与公共文本对齐。更完整的实现说明见 PERSON_2_GUIDE.md。

## 给 Person 3 及其 AI 的任务

Person 3 负责 src/models/distilbert/model.py 中的四种方法：Clean-only、Noise Augmentation、Ordinary Consistency、Entity-balanced Consistency。均从通用 distilbert/distilbert-base-cased 初始化。公共训练脚本会在 config 中提供 noisy_train_rows 和 validation_variants。按原 word index 对齐 clean/noisy，首 subword 参与监督与 JSD，忽略 padding/special tokens；长句处理、缺少实体/O 组的权重规则见 PERSON_3_GUIDE.md。用 Validation 的 Clean、Lowercase、Typo-10 平均实体 F1 选 checkpoint 和 lambda；Test 只作最终评价。

每个 model_id 分别跑 10% 和 100%：

    python -m scripts.train_model --model-id distilbert_clean --fraction 0.1 --seed 42
    python -m scripts.test_model --run-id distilbert_clean_10pct_seed42

将命令中的 model_id 依次换成 distilbert_aug、distilbert_consistency、distilbert_balanced，再将 --fraction 改为 1.0，并把测试 run_id 改为 100pct。总计 8 次训练、32 份测试评价。交付代码、测试、checkpoint、metadata、预测和结果。更完整的损失公式与伪代码见 PERSON_3_GUIDE.md。没有 GPU 的机器只做公共模块/CPU 测试，不在本机启动四种微调。

## Person 1 收集和验收

Person 1 维护公共数据、评价规则和 EXPERIMENT_MATRIX.csv。每次实际运行后检查四个条件均有 evaluation.json 和 metadata，记录训练/推理耗时与硬件；不能用 Test 重新挑参数。每位队友通过学校允许的私有方式交付需要复核的预测与日志。单独的评价汇总数字、图表和不含原文的表格可以在审核后纳入报告；带 Reuters 原文的逐句预测、errors.json/csv 不放进公开仓库。

12 次全部完成后再合并生成结果表和图，回答数据量、噪声鲁棒性、四种神经方法的差异与成本问题。某项未完成就明确标“未完成”，不要填模拟成绩。

