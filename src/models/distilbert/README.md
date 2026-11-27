# DistilBERT 微调：Person 3

四种训练目标均从通用 `distilbert/distilbert-base-cased` 初始化。Clean 使用原文 CE，Aug 平均原文和噪声 CE，普通一致性与实体平衡一致性分别再加普通 JSD 与按 gold 实体/O 分组的 JSD。每个词只有一个窗口副本参与计算，clean 与 noisy 各取其首子词并按原词顺序对齐。

默认 3 个 epoch、最大窗口 256 子词、micro-batch 1、累积 8 个原句、CUDA FP16 AMP。每个 λ 候选独立初始化并采用相同训练预算。每个 epoch 评估三个固定验证条件的严格实体 micro F1，选均值最高者。同分保留最先出现的候选和 epoch。正式 checkpoint 内的 `distilbert_run.json` 保存选择历史、实际设备及峰值分配显存。

在仓库根目录运行 `python -m scripts.run_person3`，先完成 10%，再完成 100%。使用 `--fractions 0.1` 单独运行首轮。入口调用公共训练与测试脚本，可复用已有正式 checkpoint 和完整测试结果。输入沿用共享 seed 42 数据，结果与 checkpoint 留在本地供组内复核。

本地正式运行使用 `python -m scripts.run_person3 --hyperparameters configs/person3_distilbert.json`。配置固定官方权重 revision，要求 CUDA。此前通过 `scripts.prepare_data` 生成了与组内来源一致的 seed 42 数据，源归档 SHA-256 为 `96a104d174ddae7558bab603f19382c5fe02ff1da5c077a7f3ce2ced1578a2c3`。三个划分分别为 14,041、3,250、3,453 句，并保留文档边界。

完整单元测试为 17 项通过、1 项跳过（可选 CRF 依赖）。四项 Person 3 测试已额外在 CUDA 上通过，使用人工构造样例。正式进度和四条件数值可用 `python -m scripts.summarize_person3` 从 checkpoint manifest、evaluation 和 metadata 实时读取。训练日志在 `results/person3_training.log`。

