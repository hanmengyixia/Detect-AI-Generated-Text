import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import numpy as np
import os

# 数据加载
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

# 模型定义
class BaselineLSTM(nn.Module):
    """基础模型：双向 LSTM"""
    def __init__(self, vocab_size, embed_dim, hidden_dim):
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

# 核心训练与验证
def train_and_evaluate(model, train_loader, val_loader, model_name, epochs=3, lr=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 损失函数与优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)  # 取总步数的 10% 作为预热

    # 创建余弦退火 + 预热的调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()

    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir=f'runs/{model_name}_experiment')

    best_auc = 0.0
    global_step = 0

    print(f"开始训练: {model_name}")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", position=0, leave=True, mininterval=2.0)

        for batch_idx, batch in enumerate(train_pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            # AMP 自动混合精度前向传播
            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            # 缩放反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()
            global_step += 1

            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            train_pbar.update(1)

            # 每 10 步记录一次训练 Loss 到 TensorBoard
            if batch_idx % 10 == 0:
                writer.add_scalar('Loss/Train', loss.item(), global_step)

        train_pbar.close()

        avg_train_loss = total_train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]", position=0, leave=True, mininterval=2.0)
 
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device).float().unsqueeze(1)

                with torch.cuda.amp.autocast():
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)

                val_loss += loss.item()
                # 将 Logits 转换为概率
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())

                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                val_pbar.update(1)
            val_pbar.close()

        avg_val_loss = val_loss / len(val_loader)

        # 计算验证指标
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        val_auc = roc_auc_score(all_labels, all_preds)
        val_acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))

        # 记录验证指标到 TensorBoard
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Metric/AUC', val_auc, epoch)
        writer.add_scalar('Metric/Accuracy', val_acc, epoch)

        print(f"\n--- Epoch {epoch + 1} 结果 ---")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val AUC: {val_auc:.4f} | Val Accuracy: {val_acc:.4f}\n")

        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            if not os.path.exists('saved_models'):
                os.makedirs('saved_models')
            torch.save(model.state_dict(), f'saved_models/{model_name}_best.pth')
            print(f"最佳 AUC: {best_auc:.4f}，模型已保存\n")

    writer.close()
    print(f"训练结束。最佳 AUC: {best_auc:.4f}")

# 运行入口
if __name__ == "__main__":
    # 基础配置
    ORIGINAL_DATA_PATH = "data/train_essays.csv"
    EXTERNAL_DATA_PATH = "data/train_v2_drcat_02.csv"
    MODEL_NAME = "microsoft/deberta-v3-small"
    MAX_LEN = 256
    BATCH_SIZE = 16

    # 数据融合
    print("正在加载并合并数据集")

    # 读取官方原数据
    df_orig = pd.read_csv(ORIGINAL_DATA_PATH)
    df_orig = df_orig[['text', 'generated']]

    # 读取下载的外部补充数据
    df_ext = pd.read_csv(EXTERNAL_DATA_PATH)
    df_ext = df_ext[['text', 'label']].rename(columns={'label': 'generated'})

    # 将两份数据上下拼接起来
    df_combined = pd.concat([df_orig, df_ext], ignore_index=True)

    # 打乱数据并去重 (防止外部数据集和官方数据集里有重复的文章)
    df_combined = df_combined.drop_duplicates(subset=['text']).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"合并完成 当前数据总条数: {len(df_combined)}")
    print(f"当前标签分布:\n{df_combined['generated'].value_counts()}")

    # 划分训练集和验证集 (80% 训练, 20% 验证)
    train_df, val_df = train_test_split(
        df_combined,
        test_size=0.2,
        random_state=42,
        stratify=df_combined['generated']  # 确保训练和验证集的0/1比例一致
    )

    # 加载 Tokenizer
    print("正在加载 Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 实例化 Dataset 和 DataLoader
    train_dataset = EssayDataset(train_df, tokenizer, max_length=MAX_LEN)
    val_dataset = EssayDataset(val_df, tokenizer, max_length=MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

    # 基础模型Bi-LSTM
    print("准备运行: 基础模型 (Bi-LSTM)")
    vocab_size = len(tokenizer)
    model_baseline = BaselineLSTM(vocab_size=vocab_size, embed_dim=300, hidden_dim=256)
    train_and_evaluate(
        model=model_baseline,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name="Baseline_LSTM",
        epochs=5,
        lr=1e-3
    )