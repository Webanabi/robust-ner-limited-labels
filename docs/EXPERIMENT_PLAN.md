# 实验方案

研究问题：有限标签和大小写/拼写噪声对各类 NER 模型的影响；实体平衡一致性是否优于仅增强及普通一致性；收益是否抵得上计算成本。目标“噪声 F1 上升且 Clean F1 下降不超过绝对 0.01”是待检验假设。

保留官方 train/validation/test。100% 与按句抽取 10% 是核心训练量；相同 seed 的所有模型使用同一 manifest 中 ID。抽样不按文档，交叉验证必须按 doc_id 分组。先完成 seed=42 的四种 DistilBERT，再补 CRF/LR，最后多种子和消融。30% 是开题报告的可选扩展。

模型选择：CRF 正则化与 LR/PCA 参数仅在可用训练子集内部使用 GroupKFold(n_splits=3)，groups=doc_id；任何 PCA、标准化、校准都在训练折拟合。若 doc_id=UNAVAILABLE 或不足三个真实文档，停止该 CV 并取得保留文档标记的源文件。神经网络使用 validation 的 Clean、Lowercase、Typo-10 三条件实体级 F1 的算术平均选 checkpoint 和 lambda。Test 完全不参与选择。

神经模型统一 distilbert/distilbert-base-cased 初始化，尽量保持相同原始句 batch、优化器更新预算和选择规则。Clean-only 只看原句；Aug 使用原句及副本监督 CE；Consistency 加普通 JSD；Balanced 加实体/O 各半的 JSD。完整四条件测试均使用固定公共文件。每运行保存训练秒数、推理秒数、硬件、batch size；仅硬件和 batch 等一致时直接比较延迟。

报告需要：各实体频数、句长与实体长度；模型/数据量/噪声条件/实体类别 F1；四种神经消融；Clean-to-Noisy drop；训练时间与推理成本；PCA 维度比较和二维投影（Person 2 导出特征后）；错误类别及实例。没有结果的图和表应标明未完成，不能填虚构数值。

优先级：核心四神经方法、两种训练量、seed42；随后 CRF 与原始 LR；最后特征消融、PCA、其他 seed。四神经方法的 seed123/2026 重复试验列在 repeat，CRF 消融与 PCA 在 optional。时间不足时明确在报告写明省略项，不能把简化实验写成原方案完成。

