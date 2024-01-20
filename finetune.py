import torch
from transformers import BertForSequenceClassification, BertTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], return_tensors='pt', truncation=True, padding=True, max_length=self.max_length)
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return item
    def collate_fn(self, batch):
        input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True)
        attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True)
        labels = torch.stack([item['labels'] for item in batch])

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def load_data():
    train_df = pd.read_csv("data/sst/train.csv", sep=",")
    val_df = pd.read_csv("data/sst/val.csv", sep=",")
    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    val_texts = val_df["text"].tolist()
    val_labels = val_df["label"].tolist()

    return train_texts, val_texts, train_labels, val_labels

def train():
    train_texts, val_texts, train_labels, val_labels = load_data()

    model_name = 'bert-base-uncased'
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length=128)
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length=128)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=val_dataset.collate_fn)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-08)
    num_epochs = 100
    total_steps = len(train_loader) * num_epochs

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    # Set up early stopping parameters
    early_stopping_patience = 3  
    best_val_accuracy = 0.0
    no_improvement_count = 0
    
    # Fine-tuning loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f'Train Loss: {avg_loss}')

        # Validation
        model.eval()
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}'):
                input_ids = val_batch['input_ids'].to(device)
                attention_mask = val_batch['attention_mask'].to(device)
                labels = val_batch['labels'].to(device)

                val_outputs = model(input_ids, attention_mask=attention_mask)
                val_preds = torch.argmax(val_outputs.logits.to('cpu'), dim=1).numpy()
                val_labels = labels.cpu().numpy()

                all_val_preds.extend(val_preds)
                all_val_labels.extend(val_labels)
        print(all_val_preds[:10])
        print(all_val_labels[:10])
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        print(f'Validation Accuracy: {val_accuracy}')

        # Check for early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stopping_patience:
            print(f'Early stopping after {epoch + 1} epochs without improvement.')
            break

        # Optional: Adjust learning rate
        scheduler.step()

    # Save the fine-tuned model
    model.save_pretrained("checkpoints/sst_bert")

def main():
    train()

if __name__ == '__main__':
    main()
