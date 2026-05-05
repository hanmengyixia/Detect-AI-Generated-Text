import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoConfig, AutoTokenizer

class EssayDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=256):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 提取文本和标签
        self.texts = dataframe['text'].values
        # 如果是测试集，可能没有 generated 标签
        self.labels = dataframe['generated'].values if 'generated' in dataframe.columns else None

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenizer 会将文本转换为模型可以理解的数字 ID 和 注意力掩码
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item

'''
# 模型定义
class BaselineLSTM(nn.Module):
    """基础模型：双向 LSTM"""
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # 输出 Logits

    def forward(self, input_ids, attention_mask=None):
        # LSTM 这里简化处理，忽略 attention_mask
        embedded = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(embedded)
        cat_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.fc(cat_hidden)

# 模型定义
class SotaDeberta(nn.Module):
    """SOTA 模型：标准 DeBERTa-v3"""
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # 取 [CLS] token (序列的第一个元素)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls_token)
'''

# 模型定义
class CustomDetectionModel(nn.Module):
    """自定义模型：多重池化融合网络 """
    def __init__(self, model_name="microsoft/deberta-v3-small"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        # 自定义 Head：将 CLS, Mean, Max 三种特征融合
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # 1. CLS Token
        cls_token = last_hidden_state[:, 0, :]

        # 2. Mean Pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooling = sum_embeddings / sum_mask

        # 3. Max Pooling
        last_hidden_state_masked = last_hidden_state.masked_fill(~attention_mask.bool().unsqueeze(-1), -1e9)
        max_pooling, _ = torch.max(last_hidden_state_masked, 1)

        # 特征拼接
        combined = torch.cat([cls_token, mean_pooling, max_pooling], dim=1)
        return self.fc(combined)

def generate_submission(model_path, test_csv_path, output_path, max_len=256):
    print("开始进行测试集推理并生成提交文件")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载测试数据
    test_df = pd.read_csv(test_csv_path)
    print(f"读取测试集成功！共需预测 {len(test_df)} 条数据。")

    # 加载 Tokenizer 和 Dataset
    MODEL_NAME = "/kaggle/input/datasets/chenandwen/my-detect-ai-model-1/deberta-v3-small"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_dataset = EssayDataset(test_df, tokenizer, max_length=max_len)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 加载训练好的最佳模型权重
    vocab_size = len(tokenizer)
    model = CustomDetectionModel(MODEL_NAME)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # 开始预测
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask)

            # 使用 Sigmoid 将输出转换为 0~1 的概率值
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_preds.extend(probs)

    # 生成 Kaggle 格式的提交文件
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'generated': all_preds
    })

    submission_df.to_csv(output_path, index=False)
    print(f"\预测完成！提交文件已保存至: {output_path}")

#  运行入口
if __name__ == "__main__":
    # 替换为你刚才保存的那个 0.999 AUC 的模型路径
    BEST_MODEL_PATH = "/kaggle/input/datasets/chenandwen/my-detect-ai-model-1/Custom_Deberta_best.pth"
    TEST_DATA_PATH = "/kaggle/input/competitions/llm-detect-ai-generated-text/test_essays.csv"
    OUTPUT_CSV_PATH = "submission.csv"

    generate_submission(BEST_MODEL_PATH, TEST_DATA_PATH, OUTPUT_CSV_PATH)
