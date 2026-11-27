# 实验进度

状态日期：2026-11-27。公共数据、统一训练/测试脚本、评估、汇总模块及演示骨架已建立，14 项 CPU 测试通过。CoNLL-2003 的 train/validation/test 已下载并处理到本地；三个划分句数与提案一致，文档边界可用。已完成两个真实 CRF 训练运行（10% 和 100%，均为 seed42），其他模型仍待实现。逐运行登记见 EXPERIMENT_MATRIX.csv，目前 8 行 evaluated、200 行 planned。下载来源与使用条件见 docs/DATA_PROVENANCE.md。

| 优先级 | Model ID | 训练比例 | 种子 | 状态 | 负责人 |
|---|---|---|---|---|---|
| 核心 | distilbert_clean | 100%, 10% | 42 | 待实现 | Person 3 |
| 核心 | distilbert_aug | 100%, 10% | 42 | 待实现 | Person 3 |
| 核心 | distilbert_consistency | 100%, 10% | 42 | 待实现 | Person 3 |
| 核心 | distilbert_balanced | 100%, 10% | 42 | 待实现 | Person 3 |
| 补充 | crf | 10%, 100% | 42 | 均已训练并完成四条件测试 | Person 1 起跑，Person 2 复核 |
| 补充 | lr_raw | 100%, 10% | 42 | 待实现 | Person 2 |
| 可选 | crf_no_case, crf_no_affix, lr_pca64, lr_pca128 | 100%, 10% | 42/123/2026 | 待实现 | Person 2 |
| 可选 | 四种 DistilBERT 多种子重复 | 100%, 10% | 123/2026 | 待实现 | Person 3 |

本次 crf_10pct_seed42：按真实 doc_id 三折选择 c1/c2，最佳参数 c1=0.1、c2=0.1、max_iterations=75；分组 CV 平均实体 F1 约 0.7367。用 1404 个训练句重训耗时约 11.33 秒。本机测试集严格实体级 F1：Clean 0.6899、Lowercase 0.0032、Typo-10 0.6792、Typo-20 0.6631。全部转小写后下降很大；crf_no_case 可在首批实验完成后作为可选消融复核。这是单个 seed 的结果，尚不能报告多 seed 均值/标准差。checkpoint 在 checkpoints/crf_10pct_seed42，预测和评估在 results/predictions/crf_10pct_seed42、results/metrics/crf_10pct_seed42；图表在 results/figures。以上本地大文件被 Git 忽略。

本次 crf_100pct_seed42：使用完整 14041 句、946 个原文档，按文档三折选择同样的最佳参数 c1=0.1、c2=0.1、max_iterations=75，分组 CV 平均实体 F1 约 0.8630；训练与调参合计约 123.81 秒。测试集严格实体级 F1：Clean 0.7996、Lowercase 0.0120、Typo-10 0.7803、Typo-20 0.7630。与同 seed 的 10% CRF 相比，Clean 提高约 0.1097、Typo-10 提高约 0.1011；Lowercase 仍极差。checkpoint 在 checkpoints/crf_100pct_seed42；四条件预测和评估在 results/predictions/crf_100pct_seed42、results/metrics/crf_100pct_seed42。results/tables 与 results/figures 已用两次真实运行重新汇总。

首批 12 次训练已完成 2 次 CRF，剩余 10 次：lr_raw 的 10% 与 100% 两次，以及四种 DistilBERT 方法各在 10% 与 100% 下的八次。CRF 特征消融和 PCA 不属于首批，暂不抢占这 10 次的资源与时间。

每个完成的模型/比例/种子须在 Clean、Lowercase、Typo-10、Typo-20 四个测试条件保存预测及 metadata。配置清单见 configs/experiments.yaml。其他真实结果产生后继续更新表格中的运行状态、checkpoint、预测路径和训练时间。

