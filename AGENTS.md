# 开发约定

- 当前阶段只完成 docs/TEAM_START_HERE.md 中的首批 12 个训练运行和对应 48 份测试评价，再分析结果。CRF 10% 与 100% 已完成；剩余 2 个 lr_raw 和 8 个 DistilBERT。可选消融、多 seed 与 PCA 暂缓。
- 接手本项目的 AI 先读 docs/TEAM_START_HERE.md、docs/DATA_FORMAT.md、docs/TRAIN_TEST_GUIDE.md 和自己负责的 PERSON_2_GUIDE.md 或 PERSON_3_GUIDE.md。严格区分“已实现”“待实现”和“已实际运行”。
- Person 1 管理 src/data、src/evaluation、src/analysis、src/experiment.py、configs、scripts、app.py 及接口变更。
- Person 2 主要改 src/models/crf、src/models/lr_pca，并补充自己的训练、预测脚本和测试。
- Person 3 主要改 src/models/distilbert，并补充自己的训练、预测脚本和测试。
- 共享标签顺序取自 src/data/dataset.py 和 configs/label_map.json；修改前先与 Person 1 沟通。
- 所有模型训练与预测必须遵守 src/models/base.py；输出逐句 JSONL，原始 word 粒度标签。
- 任何模型不得用测试集挑参数；PCA 必须每个 CV 折内拟合；实验状态按真实进度记录。
- data/raw、data/processed、data/splits、checkpoints、results 中的敏感数据与大文件不提交 GitHub。
- 无 GPU 机器仅运行公共模块与 CPU 测试；不得把人工样例成绩写入真实结果表。

