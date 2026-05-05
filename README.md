# Detect-AI-Generated-Text
本项目旨在通过深度学习技术识别文本是由 **人类撰写** 还是由 **大型语言模型 (LLM)** 生成。本项目包含了从基础的循环神经网络到先进的 Transformer 架构的多种实现方案，专门针对 Kaggle 竞赛环境进行了优化。

## 📂 文件说明

| 文件名 | 描述 |
| :--- | :--- |
| `train_lstm.py` | **基准模型**。使用双向 LSTM 结合嵌入层，适合快速迭代和轻量化部署。 |
| `train_deberta.py` | **Transformer 模型**。基于 `DeBERTa-v3-small` 的标准微调脚本。 |
| `train_custom_model.py` | **自定义进阶模型**。引入了多重池化融合（CLS + Mean + Max）的 DeBERTa 模型。 |
| `test.py` | **推理脚本**。加载训练好的模型权重，并生成 `submission.csv`。 |

## 🛠️ 核心架构：特征融合策略

在 `train_custom_model.py` 中，我们不仅使用了 Backbone 的输出，还自定义了头部（Head）来增强特征提取：

1. **Mean Pooling**: 捕捉序列整体的平均语义。
2. **Max Pooling**: 捕捉序列中最显著的特征点。
3. **CLS Token**: 代表句子的全局表征。



## ⚙️ 快速开始

### 环境依赖
```bash
pip install torch transformers pandas scikit-learn tqdm tensorboard
