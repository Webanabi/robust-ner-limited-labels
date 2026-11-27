# 训练与测试命令

这两个公共脚本已经实现并通过 CPU 测试。CRF 已调用 sklearn-crfsuite 实现并完成 10% 与 100%（seed42）两次真实运行；LR/PCA、DistilBERT 模型的 train/save/load/predict 仍由 Person 2、Person 3 实现。未实现的模型命令会明确抛出 NotImplementedError，不会产生虚假的成绩。

## 训练一个模型

在项目根目录运行，例如 10% 数据、seed 42 的 CRF（当前已有同名 checkpoint，重复实验请增加 --run-id 新名称）：

    python -m scripts.train_model --model-id crf --fraction 0.1 --seed 42

四种神经方法只需将 --model-id 换成 distilbert_clean、distilbert_aug、distilbert_consistency 或 distilbert_balanced。完整训练集用 --fraction 1.0。默认 run_id 为“模型名_10pct_seed42”等形式，可用 --run-id 自定义。可加 --hyperparameters 参数指向只含模型超参数的 JSON/YAML 文件。若用 seed 123/2026，先重新运行 prepare_data --seed 123 或 --seed 2026 生成对应共享子集和噪声副本。

脚本先读 data/processed 中的训练文件、10% 或 100% manifest、对应噪声副本及三个固定验证条件。ID、词数、标签或种子不符就停止。模型的 train 接收原句、完整验证集和 config；config 中含 noisy_train_rows 与 validation_variants，队友可直接使用。模型成功 train/save 且 checkpoint 目录有文件后，才写 checkpoints/<run_id>/manifest.json，记录训练秒数、硬件、超参数与数据哈希，并把 EXPERIMENT_MATRIX.csv 中匹配的计划行标为 trained。模型失败时不宣称训练成功。

模型内部必须按实验方案做参数选择：CRF/LR 在训练子集里按文档三折；神经模型按验证集 Clean、Lowercase、Typo-10 的平均实体 F1。测试集不传给 train。

## 测试一个已训练模型

例如：

    python -m scripts.test_model --run-id crf_10pct_seed42

脚本读取该目录的 trained manifest 和模型权重，依次在固定的 test_clean、test_lowercase、test_typo10、test_typo20 文件上预测。每个条件输出：

- results/predictions/<run_id>/test_<condition>.jsonl：逐句原词级标签。
- 同目录 test_<condition>_metadata.json：训练/推理时间、硬件、参数。
- results/metrics/<run_id>/test_<condition>/evaluation.json、evaluation.csv、errors.json、errors.csv：严格实体级结果与错误案例。
- results/metrics/<run_id>/test_summary.json：四条件摘要。

每个条件评估成功后，矩阵对应行才标为 evaluated。若只想检查验证集，可加 --split validation；验证运行不会修改测试进度。--conditions clean typo10 可选择部分条件。推理时实际使用 CPU/GPU 由模型实现决定，metadata 的 inference_device 应由模型的 device 属性提供；只看“CUDA 可用”不能证明模型真的在 GPU 上运行。

## 验证脚本

    python -m unittest discover -s tests -v

tests/test_training_scripts.py 使用明确标记的临时测试模型，在系统临时目录跑完整训练→四条件预测→评估→矩阵更新。它不在项目 results 中生成成绩。真实模型实现后，队友还需分别做 save/load 一致性、长句与 subword 对齐测试。当前本机无需 GPU。

