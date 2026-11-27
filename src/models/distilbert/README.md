# DistilBERT 微调：Person 3 待实现

实现 clean、noise augmentation、ordinary consistency、entity-balanced consistency 四种训练方式；保存和加载真实模型，按原词级导出预测。首 subword、不同切分的词级对齐、长句窗口和稳定 JSD 细节见 docs/PERSON_3_GUIDE.md。当前 model.py 抛 NotImplementedError。

