# 数据和预测协议

每行 UTF-8 JSON。句子字段：id（稳定唯一）、doc_id、tokens（原始单词序列）、ner_tags（同长 IOB2 标签）。预测字段：id、tokens、pred_tags（原始单词级）；不得交 subword 标签。标签顺序：O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC。Gold 必须合法 IOB2；预测可含非法 I-X，但公共评估先统一修复为 B-X。

原始 CoNLL 文件可能采用 IOB1：实体首词会写成 I-X。本地读取程序会将不能接续前词的 I-X 转为 B-X，并在每句 source_tag_conversions 与统计文件中记录转换次数。处理后所有成员只使用统一 IOB2。

噪声数据保留原 id、doc_id、ner_tags，增加 source_id 和 condition。ID 匹配时 token 必须与该条件的公共 gold JSONL 完全相同。训练副本由 training_pair 按 id 与 seed 确定；验证与测试使用独立种子。lowercase 为全部小写；typo10/20 按每个合格词独立 Bernoulli 选择，具体实际比例由 audit 文件记录。训练时两类扰动等概率选择，每原句生成一个副本。

每次真实实验需要一份 metadata.json，必含 model_id、fraction、seed、condition、training_seconds、inference_seconds、hardware、hyperparameters；建议再加 git_commit、dataset_source、checkpoint、batch_size、tokenizer_revision、torch_version。时间单位为秒。训练时间可在同一 seed 的四种测试条件重复记录，但汇总成本时只计一次。

模型预测写出示例：

    {"id":"test_000001","tokens":["I","live","in","London","."],"pred_tags":["O","O","O","B-LOC","O"]}

公共 API：src.data.dataset.read_jsonl/write_jsonl/repair_iob2/spans；src.data.sampling.sample_sentences；src.data.noise.perturb/training_pair；src.evaluation.evaluate.evaluate_files。

