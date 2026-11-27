# 冻结 DistilBERT + LR/PCA：Person 2 待实现

实现 lr_raw、lr_pca64、lr_pca128。使用通用 distilbert/distilbert-base-cased 的冻结词向量，首 subword 映射回原词。PCA 与 LR 在 GroupKFold 每折内拟合。参见 docs/PERSON_2_GUIDE.md。当前 model.py 抛 NotImplementedError。

