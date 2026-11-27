# Person 2：CRF 与冻结特征基线

## 要做什么、为什么

实现三套 CRF（完整、去大小写、去词缀）和三套冻结 DistilBERT + Logistic Regression（原维、PCA64、PCA128）。CRF 检查手工词形特征与传统序列约束；冻结表示组检查预训练上下文表示的价值以及 PCA 的效率/精度权衡。CRF 公共版本已经调用 sklearn-crfsuite 实现，并完成 crf 的 10%/seed42 一次真实运行；你负责复核与扩展消融、完成 LR/PCA。先理解 IOB2、CRF 线性链序列建模、词级向量、PCA、分组交叉验证和数据泄漏。

## 文件和依赖

修改 src/models/crf/model.py、src/models/lr_pca/model.py，完善各目录 README 并编写必要测试。公共 scripts/train_model.py 和 scripts/test_model.py 已提供统一入口。基类是 src/models/base.py，模型 ID 在 registry.py。建议 sklearn-crfsuite、scikit-learn、transformers、torch；安装 requirements-gpu.txt（CRF 单独用 CPU 可只装 sklearn-crfsuite）。参考 sklearn-crfsuite 文档 https://sklearn-crfsuite.readthedocs.io/ 和 scikit-learn GroupKFold https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html 、Pipeline/PCA https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html 。Transformer 的 word_ids 对齐见 https://huggingface.co/docs/transformers/tasks/token_classification 。

## CRF 实现顺序

1. 从 read_jsonl 读取同 seed 的 train_10pct_seed42 和完整 validation。每句只使用 tokens/ner_tags；绝不使用 POS 或 chunk。为每词造字典特征：词本身/小写、大小写状态、数字模式、前后邻词、2-4 字母前后缀、句首句尾。消融时分别去掉所有 casing 派生键或所有 prefix/suffix 键，其他保持不变。
2. 用 sklearn_crfsuite.CRF 拟合序列标签。训练前根据 doc_id 构建 GroupKFold(3)，每折文档不交叉，调 c1/c2 等正则化。若 doc_id=UNAVAILABLE 或真实文档不足三组，明确报错并向 Person 1 请求保留 -DOCSTART- 的文件，不按句假装文档 CV。
3. 在完整可用训练子集上用选好的参数重训。predict 返回每原始词一个标签，经过 predict_checked；save/load 保存特征配置、CRF、版本信息，避免加载不同消融的参数。测试空/标点/大小写输入。

## 冻结特征 + LR/PCA

1. 用 distilbert/distilbert-base-cased 的通用预训练权重，encoder.eval 且 torch.no_grad；不可使用 NER 微调 checkpoint。tokenizer 输入 is_split_into_words=True，利用 word_ids 找每个原始词的首个 subword 表示；special/padding 不取。长句须窗口化并让每词恰有一个向量。训练和推理使用完全相同的抽取方式。
2. 训练 LR 原维特征；PCA64/128 分别作为独立配置。每个 GroupKFold 的训练折内部拟合 PCA（及任何 scaler），再变换验证折。不能在全部训练子集上先 fit PCA 后做 CV，更不能用官方 validation/test fit。
3. 选择 LR 的 C 和 PCA 维度组合时按训练内文档分组 CV。选好后全训练子集拟合特征变换与分类器。预测原始词级标签，经统一 BIO 修复后导出；保存 tokenizer/encoder 标识、PCA、LR、标签顺序、版本。提供 PCA 二维投影用的训练子集特征与标签供 Person 1 作图，但不得把可还原语料公开上传。

## 统一输入、输出与运行

每个配置实现 NERModel.train(train_rows, validation_rows, config)、predict(tokens)、save(path)、load(path)。尽量让 model_id 与配置对应，必要时由 manifest 标明 crf_no_case 等。公共训练命令会把原始训练句、噪声副本和验证条件传给模型；你的 CRF/LR 可忽略噪声副本，但必须在训练子集内按文档分组选择参数。待你实现接口后，用 scripts/train_model.py 训练、scripts/test_model.py 测四条件；当前尚无可运行的真实模型训练或推理。预测必须写 JSONL 的 id/tokens/pred_tags，长度与公共 test_*.jsonl 一致。每个模型/比例/seed/condition 单独存 metadata.json，记训练秒数、推理秒数、硬件、超参数、checkpoint。

运行后 Person 1 用公共命令评估，例如：

    python -m scripts.run_evaluation --gold data/processed/test_typo10.jsonl --predictions results/predictions/CRF_TYPO10.jsonl --metadata results/predictions/CRF_TYPO10_metadata.json --output results/metrics/CRF_TYPO10

自测至少包括：词级标签与 tokens 同长；三种 CRF 消融真的改变对应特征；PCA fit 仅在折内；同一文档不跨训练/验证折；save/load 后预测一致；重复 seed 结果与 manifest 一致；四条件均能通过公共评估。

## 交付和取舍

先做 CRF 与 lr_raw 在 10%/100%、seed42 下的四条件真实运行，再做 CRF 消融与 PCA64/128，最后补 seed123/2026。必交模型代码、训练/推理命令、checkpoint（受控共享）、预测 JSONL、metadata、调参记录、方法/错误分析文字。若时间不足可推迟 PCA 二维图和多 seed，但不把未跑项目写成完成。自己的部分完成后可协助 Person 3 跑实验，保持其配置不可私自改变。

