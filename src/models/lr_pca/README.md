# 冻结 DistilBERT + Logistic Regression

Person 2 的首批实验使用 `lr_raw`，在 seed 42 下分别训练 10% 和 100% 两个模型。PCA64 和 PCA128 留待后续实验。

## 方法

使用通用预训练模型 `distilbert/distilbert-base-cased`，固定版本见 `encoder.py` 的 `MODEL_REVISION`。编码器在 eval 和 inference 模式下抽取每个原词首个 WordPiece 的隐藏表示。超过上下文长度的输入用带重叠的窗口处理，每个原词只保留一次表示。分词器规范化后消失的词使用一个 UNK。训练与预测复用同一套抽取逻辑。

以训练子集的真实文档 ID 为组做三折 GroupKFold。每个候选 C 在折内训练 LR，对留出文档逐句修复 IOB2，然后计算严格实体级 micro F1。按三折 F1 的算术平均选 C，同分取候选列表中靠前者，再用整个训练子集重训。默认候选为 0.1、1、10，求解器为 lbfgs，最多迭代 300 次。正式验证集和测试集只通过公共脚本用于其各自的用途。

## 运行与复核

从仓库根目录运行。在虚拟环境安装 `requirements-gpu.txt` 后，CPU 即可运行。首次训练需下载固定版本的编码器。将课程允许使用、含文档边界的三份 CoNLL 文件置于 `data/raw/`。

```sh
python -m unittest discover -s tests -v
python -m scripts.prepare_data --source local --train data/raw/train.txt --validation data/raw/valid.txt --test data/raw/test.txt --seed 42
python -m scripts.train_model --model-id lr_raw --fraction 0.1 --seed 42
python -m scripts.test_model --run-id lr_raw_10pct_seed42
python -m scripts.train_model --model-id lr_raw --fraction 1.0 --seed 42
python -m scripts.test_model --run-id lr_raw_100pct_seed42
```

`checkpoints/<run_id>/manifest.json` 记录训练耗时，`training_report` 包含各候选的三折分数、最终参数、设备与库版本。该目录包含重载所需的编码器、分词器和 LR 分类器，可离线预测。特征抽取默认批大小为 8。可使用公共训练入口的 `--hyperparameters` 设置 `batch_size`、`device`、`c_values` 和 `max_iter`。

四条件的总体指标见 `results/metrics/<run_id>/test_summary.json`，推理耗时见 `results/predictions/<run_id>/test_<condition>_metadata.json`。条件名为 `clean`、`lowercase`、`typo10` 和 `typo20`。逐句预测、错误案例、语料及权重仅在受控环境交接。
