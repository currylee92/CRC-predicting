from transformers import BertForSequenceClassification, BertTokenizer, AdamW
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# 定义数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 定义训练BERT模型的函数
def train_bert_model(texts, labels, model_name='bert-base-chinese', max_length=128, batch_size=16, epochs=2):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)

    # 将数据划分为训练集和验证集
    texts_train, texts_val, labels_train, labels_val = train_test_split(texts, labels, test_size=0.2)

    # 创建数据加载器
    train_dataset = TextDataset(texts_train, labels_train, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataset = TextDataset(texts_val, labels_val, tokenizer, max_length)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 定义优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        # 验证循环
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss = outputs.loss

        print(f'Epoch {epoch+1}/{epochs} - Train loss: {loss.item()} - Val loss: {val_loss.item()}')

    return model
