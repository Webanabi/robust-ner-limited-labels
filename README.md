# Robust NER with Limited Training Labels

本项目比较 CoNLL-2003 上的 CRF、冻结 DistilBERT 表示加 Logistic Regression、以及四种 DistilBERT 微调目标。公共数据处理、噪声生成、实体级评估、结果汇总已实现。CRF 已调用 sklearn-crfsuite 完成 10% 和 100%（seed42）的真实训练和四条件测试；LR/PCA 与 DistilBERT 仍由 Person 2、Person 3 在对应骨架中实现。

**队友及其 AI 请先读 docs/TEAM_START_HERE.md。**当前只推进首批 12 个训练运行和 48 份固定条件测试；其中 CRF 的 2 个训练运行已完成，其余 10 个待完成。实验完成后再集中分析。

第一次上传 GitHub 和邀请队友的步骤见 docs/GITHUB_UPLOAD.md。

## 最快启动

在项目根目录创建 Python 3.10+ 虚拟环境并安装 requirements.txt。PowerShell 命令为：

    py -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v

获得合法的 CoNLL 原始文件后，保存在 data/raw（已被 Git 忽略）。原始文件必须包含 -DOCSTART- 行，才可进行真实按文档分组交叉验证。运行：

    python -m scripts.prepare_data --source local --train data/raw/train.txt --validation data/raw/valid.txt --test data/raw/test.txt

当前工作区已经准备好这三个本地文件并运行过上述命令。若队友从 GitHub 克隆仓库，需自行按课程许可取得数据；原始和处理后的文本不会随仓库上传。来源及校验见 docs/DATA_PROVENANCE.md。

输出保存在 data/processed。脚本会检查三个划分的句数与开题报告所述的 14041、3250、3453 是否相符，记录实际数量而不强行修改。也可使用 --source hf，但该源的 doc_id 标记为 UNAVAILABLE；Person 2 的按文档分组 CV 需换用保留边界的原始文件。

数据准备、噪声概率和默认 seed 读取 configs/default.yaml；可用 --seed、--fraction 覆盖，并重复运行生成其他共享子集。模型实现后，scripts/predict.py 按统一接口输出预测与推理时间。

模型导出单词级预测后，执行：

    python -m scripts.run_evaluation --gold data/processed/test_clean.jsonl --predictions results/predictions/RUN.jsonl --metadata results/predictions/RUN_metadata.json --output results/metrics/RUN

统一训练与四条件测试入口已实现。CRF 的 CPU 环境可安装 requirements-crf.txt，命令为 python -m scripts.train_model --model-id crf --fraction 0.1 --seed 42，随后运行 python -m scripts.test_model --run-id crf_10pct_seed42。10% 和 100% 的同名运行均已完成；重复实验需指定新 run_id。详细文件与结果位置见 docs/TRAIN_TEST_GUIDE.md。LR/PCA 与 DistilBERT 骨架仍会明确抛出 NotImplementedError。

在至少一个真实 evaluation.json 存在后：

    python -m scripts.generate_report_assets --evaluations results/metrics/RUN/evaluation.json --statistics data/processed/train_statistics.json data/processed/validation_statistics.json data/processed/test_statistics.json

EXPERIMENT_MATRIX.csv 列出每个计划运行的比例、种子、条件、方法、状态、模型和预测路径。它由 python -m scripts.init_experiment_matrix 从配置生成；更新进度时直接编辑已有 CSV，命令默认不会覆盖。

app.py 通过 streamlit run app.py 启动。已训练的 CRF 可通过 manifest 被选中；尚未实现的 LR/PCA 和 DistilBERT 会明确抛出 NotImplementedError。

详见 docs/DEVELOPMENT_GUIDE.md；各成员先看对应 PERSON_*_GUIDE.md。

