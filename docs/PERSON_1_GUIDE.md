# Person 1：公共模块与集成

## 目标与已完成部分

你的任务是让不同模型共享同一实验事实：相同句子、标签、子集、噪声、指标和报告入口。已实现 src/data、src/evaluation、src/analysis 的初版、scripts 命令、app.py 骨架和 CPU 测试；你负责继续核对真实数据、检查模型交付和完善图表。无需在本机训练 GPU 模型。概念上需理解 IOB2、严格实体 span F1、数据泄漏、文档分组 CV 与随机种子。

## 你维护的文件及输入输出

- configs/default.yaml、experiments.yaml、label_map.json：统一设置；变更时通知两位队友。
- src/data/dataset.py：JSONL 校验、标签映射、BIO 修复、实体抽取，输入句子/预测 dict，输出合法性或半开区间 span。
- src/data/preprocess.py：本地 CoNLL 与备选 HF 读取，输入官方 split 文件，输出句子 dict；真实文档边界只能由 -DOCSTART- 取得。
- src/data/sampling.py：按 seed 抽 10%，输出原序子集和 ID/数量清单。
- src/data/noise.py：按 ID/seed/条件生成词数不变的副本和 audit；训练副本保留 gold 标签。
- src/data/statistics.py：实体数、类型、句长、实体长度。
- src/evaluation：验证预测覆盖、严格 F1、互斥错误归类，输出 evaluation.json、errors.json/csv。
- src/analysis：读取真实 evaluation.json，输出 runs/summary/drops/costs CSV 和训练成本、神经消融、错误统计等图；二维 PCA 图依赖 Person 2 的导出数据。
- app.py：按 trained manifest 加载模型，显示实体、原文偏移、干净/噪声对比并导出。要用 HTML 安全转义和颜色高亮；当前应用骨架可提示尚无模型。

## 操作步骤

1. 核对学校许可，取得保留 -DOCSTART- 的三个官方源文件。运行 README 的 prepare_data 命令，检查 source_audit 与原提案句数差异，手工对照几处文档边界与标签。若只用 HF，明确告知 Person 2 不能做真正的按文档 CV。
2. 把 10% manifest 的 ID 和对应公共文件提供给两位队友；他们不可自行抽样。Validation/Test 固定噪声文件也提供给两人。检查 audit 的 changed、eligible 总数和数条人工样例。
3. 让队友先用 tests 中的小样例熟悉预测格式，再检查其第一份真实 predictions 是否通过 run_evaluation。核对 metadata 与训练日志、checkpoint 路径。
4. 收齐真实运行后执行 generate_report_assets，检查图中缺失的模型/seed，更新 EXPERIMENT_STATUS.md；依据研究问题写分析，不把相关性写成因果结论。
   若 Person 2 提供二维 PCA 投影 CSV（列 x,y,label），给命令增加 --pca-points 路径。
5. 启动 Streamlit 验证已训练模型的输入、实体类别/字符偏移、噪声对比、JSON/CSV。与原始文本核对 offset；对 CRF 和神经模型均尝试。

## 命令和验证

    python -m unittest discover -s tests -v
    python -m scripts.prepare_data --source local --train data/raw/train.txt --validation data/raw/valid.txt --test data/raw/test.txt
    python -m scripts.run_evaluation --gold data/processed/test_clean.jsonl --predictions results/predictions/RUN.jsonl --metadata results/predictions/RUN_metadata.json --output results/metrics/RUN
    python -m scripts.generate_report_assets --evaluations results/metrics/RUN/evaluation.json
    streamlit run app.py

至少核对：标签顺序、10% 重复运行相同 ID、扰动不改词数/标签、非法预测检测、BIO 修复、实体边界 F1、四种互斥错误、空结果不生造指标、报告图来自真实文件。测试中小样例只用于工程验收。

## 交付与优先级

必做：合法真实数据与 audit、公共测试、队友预测验收、真实结果汇总、项目整合、报告和演示。若时间不足，优先服务四种神经核心实验；PCA 二维图、额外 seed、外观优化可延后，但状态表如实标出。最终交给组内：公共 JSONL 与子集 manifest 的受控副本、统一配置、脚本命令、评价表和图、错误案例、复现说明。

参考：开题报告；CoNLL 数据卡 https://huggingface.co/datasets/eriktks/conll2003；Hugging Face token classification 教程 https://huggingface.co/docs/transformers/tasks/token_classification 。文档中具体 split 与使用条款以实际下载源和课程要求为准。

