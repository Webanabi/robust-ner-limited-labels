# Person 3：DistilBERT 四种训练方法

## 目标与概念

设备：八次微调优先在 Person 3 的 RTX 3060 Laptop 上运行。该型号标准显存为 6 GB，但先用 nvidia-smi 查看实际容量，再用 torch.cuda.is_available() 确认 PyTorch 可见 GPU。若为 6 GB，从 10% 数据和 micro-batch=1 开始测显存/耗时，用梯度累积保持目标有效 batch；必要时尝试混合精度、缩短窗口和梯度检查点。原文与扰动副本并用的目标可能比 Clean-only 更占显存。试跑后四种方法尽量固定同一窗口长度和有效 batch。若 batch=1 仍无法稳定运行，学校的单张 T4（16 GB 显存）是备用资源，需确认授权与配额。记录每种方法的真实设备、batch、窗口长度、耗时，避免跨设备直接比速度。

从通用 distilbert/distilbert-base-cased 预训练权重开始，做四种可比的词级 NER 微调：distilbert_clean、distilbert_aug、distilbert_consistency、distilbert_balanced。要回答实体平衡损失是否超越单纯噪声增强与普通一致性，同时检查 Clean F1 是否受损。当前只有 src/models/distilbert/model.py 的 NotImplementedError 骨架；你要实现训练、预测、保存、加载与实验脚本。需要理解交叉熵、softmax、Jensen-Shannon divergence、subword/word_id 对齐、mask、梯度累积和长句滑窗。

依赖 torch、transformers、accelerate，可用 Hugging Face Token Classification 教程 https://huggingface.co/docs/transformers/tasks/token_classification 和课程 https://huggingface.co/docs/course/chapter7/2 作为对齐参考。禁止使用已经在 CoNLL-2003 上训练过的 NER 模型。绝不使用 POS/chunk 作为输入。

## 输入输出与文件

修改 src/models/distilbert/model.py，完善目录 README 并编写测试。公共 scripts/train_model.py 和 scripts/test_model.py 已提供统一入口；train 的 config 中有 noisy_train_rows 和 validation_variants。使用 src.data.dataset 的标签映射与 BIO 修复；绝不在自己代码中重新抽取 10% 或重做测试噪声。train 接受训练/验证句子和 config；predict(tokens) 返回原始词级 BIO，长度等于 tokens；save/load 必须保存权重、tokenizer/映射、超参和最佳 checkpoint 信息。模型类的 model_id 应与实际四种方法一致，或在 manifest 中明确配置。

每运行写预测 JSONL id/tokens/pred_tags 与 metadata.json（model_id、fraction、seed、condition、training_seconds、inference_seconds、hardware、hyperparameters、checkpoint）。对公共 test_clean/lowercase/typo10/typo20 分别推理，不能自行临时生成另一版测试噪声。

## 四个目标函数

对原句 x、一个固定或按 epoch 明确生成的扰动副本 x' 和同一词级 gold y：

L_sup = [CE(x,y) + CE(x',y)] / 2。

Clean-only 只计算 CE(x,y)。Noise Augmentation 使用 L_sup。Ordinary Consistency 使用 L_sup + λ L_all。Entity-balanced Consistency 使用 L_sup + λ L_bal。λ 从 0.1、0.5、1.0 中按 Validation 选择。上述 CE 是有效原始词的平均，不是所有 subword 的平均。训练副本每原句一种扰动，lower 与 typo 等概率；下采样由公共文件提供。

对每个原始词 i，从两个编码分别拿首个 subword 的 logits，p_i=softmax(clean_logits_i)，q_i=softmax(noisy_logits_i)，m_i=(p_i+q_i)/2。J_i = [KL(p_i||m_i)+KL(q_i||m_i)]/2。用 log_softmax / logsumexp 或 clamp(min=1e-8) 避免 log(0)，保持梯度通过两支，不能 detach 任一预测分布。

普通损失 L_all = mean(J_i | valid word i)。平衡损失先按真实 y_i 分组：E={i:y_i≠O}，N={i:y_i=O}。若两组都有词，L_bal=0.5 mean(J_E)+0.5 mean(J_N)；只有一组非空时，L_bal=该组 mean；不计 padding 和 special token。分组依据 gold，不依赖模型预测。batch 的两个句子可产生不同 subword 长度，必须先按原始 word index 配对，不能按 subword 位置直接相减。

可实现为以下逻辑（伪代码，非现成训练程序）：

    clean_logits = model(clean_encoding).logits
    noisy_logits = model(noisy_encoding).logits
    clean_words = first_subword_by_word_id(clean_encoding, clean_logits)
    noisy_words = first_subword_by_word_id(noisy_encoding, noisy_logits)
    assert clean_words.keys() == noisy_words.keys() == gold_word_ids
    ce_clean = cross_entropy(clean_words, gold_tags)
    ce_noisy = cross_entropy(noisy_words, gold_tags)
    jsd = stable_jsd(softmax(clean_words), softmax(noisy_words))
    entity_mask = gold_tags != LABEL_TO_ID["O"]
    ordinary = jsd.mean()
    balanced = mean_present_groups(jsd[entity_mask], jsd[~entity_mask])
    loss = (ce_clean + ce_noisy) / 2 + lambda_value * selected_consistency

mean_present_groups 在双组存在时各 0.5，单组时该组 1.0。忽略的词不能加入分母。先在小型构造张量测试：两组齐全、全 O、全实体、概率相等时 JSD≈0、反向传播有有限梯度、不同 subword 切分仍能对齐。

## 对齐、长句与训练公平性

tokenizer(tokens, is_split_into_words=True) 后用 word_ids 找每个 word 的首个 subword；后续 subword、padding、special token 的监督标签为 -100。clean 与 noisy 各自编码，分别建立 word index → logits 映射。截断不能悄悄丢掉词：用重叠窗口，给每个原始词唯一的最终预测（例如选择离窗口边缘更远的覆盖位置）；训练时也须明确定义窗口损失去重，不能重复计算边缘词。推理回到原始词序、修复非法 I-X，再导出。

四种方法的原始句 batch、通用 checkpoint 初始化、优化器和更新次数预算应一致。增强/一致性需要额外前向计算，记录每方法训练耗时与显存条件。每个 seed 新建模型；别从前一方法微调权重接着训。显存不足可降低 micro-batch 并增加 gradient accumulation，使有效 batch 保持近似一致，记录任何差异。

## 选择、运行、验证

仅用 Validation 的 Clean、Lowercase、Typo-10 实体级 micro F1 平均挑 checkpoint 与 λ；不要用 Test 或 Typo-20 挑选。公共 eval API 可对验证预测文件评估，记录三个 F1 的平均。用 seed42 先跑四方法在 10% 训练量，再扩展 100%；然后考虑 seed123/2026。统一命令见 TRAIN_TEST_GUIDE.md；只有你实现真实 train/save/load/predict 后才可运行。

至少验证：同一词的 clean/noisy 标签对应；第一 subword 规则；mask 后的词数等于原句词数；CE 和 JSD 维度/数值有限；缺一组时权重 100%；save/load 预测相同；长句每词恰有一次预测；预测 JSONL 能通过公共评估；所有模型共享 10% manifest 与四种固定测试文件。公共 scripts/predict.py 可在你实现 load/predict 后统一导出并计时。

交付：代码和单元测试、训练/预测命令、checkpoint（受控共享）、每运行配置/日志/metadata/预测、参数选择记录、四方法比较与负面结果说明、报告方法部分。若时间不足，先确保四方法 seed42、10%/100% 的核心对比；其他 seed 和更多 λ 尝试需明确记为未完成，不推定改进方法会获益。

