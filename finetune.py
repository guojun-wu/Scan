import torch
from transformers import ( 
    AutoModelForSequenceClassification,
    AutoTokenizer, 
    GPT2Tokenizer,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, task):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        if task == 'sst':
            self.label_mapping = {0: 0, 2: 1, 4: 2}
        elif task == 'wiki':
            self.label_mapping = {'award': 0, 'education':1, 'employer':2, 'founder':3, 
                        'job_title':4, 'nationality':5, 'political_affiliation':6, 'visited':7, 'wife':8}
        else:
            raise ValueError("Invalid task")
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], return_tensors='pt', truncation=True, padding=True, max_length=self.max_length)
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.label_mapping[self.labels[idx]], dtype=torch.long)
        }
        return item
    def collate_fn(self, batch):
        input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True)
        attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True)
        labels = torch.stack([item['labels'] for item in batch])

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def load_data(task):
    train_df = pd.read_csv(f"data/{task}/train.csv", sep=",")
    val_df = pd.read_csv(f"data/{task}/val.csv", sep=",")
    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    val_texts = val_df["text"].tolist()
    val_labels = val_df["label"].tolist()

    return train_texts, val_texts, train_labels, val_labels

def train(model_name, model, tokenizer, task):
    train_texts, val_texts, train_labels, val_labels = load_data(task)

    train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length=128, task=task)
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length=128, task=task)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=val_dataset.collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-08)
    num_epochs = 10
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

        scheduler.step()

    # Save the fine-tuned model
    model.save_pretrained(f"checkpoints/{task}_{model_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model_name', type=str, default='bert')
    parser.add_argument('-t','--task', type=str, default='sst')
    args = parser.parse_args()
    model_name = args.model_name
    task = args.task
    model_dict = {
        "distilbert": "distilbert-base-uncased",
        "bert": "bert-base-uncased", 
        "roberta": "roberta-base", 
        "gpt2": "gpt2", 
        "opt": "facebook/opt-350m",
        "bert_large": "bert-large-uncased",
        "gpt2_large": "gpt2-large",
    }
    if task == 'sst':
        num_labels = 3
    elif task == 'wiki':
        num_labels = 9
    if model_name in model_dict:
        model = AutoModelForSequenceClassification.from_pretrained(model_dict[model_name], num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(model_dict[model_name])
    # if model_name contains 'gpt2', then use GPT2Tokenizer
    else:
        raise ValueError("Invalid model name")
    if 'gpt2' in model_name:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        padding_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
        model.config.pad_token_id = padding_token_id
        tokenizer.pad_token_id = padding_token_id
    train(model_name, model, tokenizer, task)
    

if __name__ == '__main__':
    main()
