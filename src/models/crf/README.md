# CRF 基线

已用 sklearn-crfsuite 实现 crf、crf_no_case、crf_no_affix 的 train/predict/save/load。特征只取 tokens，不用 POS/chunk。c1/c2 正则化在训练子集内按真实 doc_id 三折选择，再以最佳参数重训。默认候选为 (0,0.1)、(0.1,0.1)、(0.1,1.0)，默认最大迭代 75；可通过训练命令的 --hyperparameters JSON/YAML 文件覆盖。

训练与四条件测试命令见 docs/TRAIN_TEST_GUIDE.md。模型权重由 joblib 保存，只应加载来自可信来源的 checkpoint。Person 2 可继续调整特征、候选范围和消融实验，任何改动应记录在配置及结果中。

