# 本地数据来源与检查

2026-11-27，从 Hugging Face eriktks/conll2003 的数据加载脚本指向的 https://data.deepai.org/conll2003.zip 下载到 data/raw/conll2003.zip，仅解压 train.txt、valid.txt、test.txt 到 data/raw。压缩包 SHA-256：96a104d174ddae7558bab603f19382c5fe02ff1da5c077a7f3ce2ced1578a2c3。

已运行 scripts.prepare_data。检查结果：

| 划分 | 句子 | 文档 | IOB1→IOB2 转换次数 |
|---|---:|---:|---:|
| train | 14041 | 946 | 0 |
| validation | 3250 | 216 | 0 |
| test | 3453 | 231 | 0 |

此归档已采用 IOB2，并保留 -DOCSTART-；三个划分的句数均与开题报告吻合。seed 42 的 10% 训练子集有 1404 句、19006 词、2259 个实体。测试集 typo10 实际合格词扰动率约 9.92%，typo20 约 20.14%。完整哈希、统计和噪声审计在 data/processed/source_audit.json 与同目录文件中。

数据卡：https://huggingface.co/datasets/eriktks/conll2003 。该卡把许可证标为 other，并说明英文文本源于 Reuters。NIST 的 Reuters 页面说明其研究使用受机构和个人协议约束：https://trec.nist.gov/data/reuters/reuters.html 。下载到本地并不代替核实学校是否已有适用授权；请不要把 data/raw、data/processed 或可还原文本的预测实例提交到公开 GitHub。两目录已列入 .gitignore。

