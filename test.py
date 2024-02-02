from finetune import *

# Load the model
def load_test_data(task):
    test_df = pd.read_csv(f'./data/{task}/test.csv', sep=",")
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()
    return test_texts, test_labels

def test(model, tokenizer, task):
    test_texts, test_labels = load_test_data(task)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length=128, task=task)
    test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=test_dataset.collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    all_preds = []
    all_labels = []
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test accuracy: {accuracy:.4f}")

def load_model(model_name, task):
    # load from checkpoint f"checkpoints/{task}_{model_name}"
    model_dict = {
        "distilbert": "distilbert-base-uncased",
        "bert": "bert-base-uncased", 
        "roberta": "roberta-base", 
        "gpt2": "gpt2", 
        "opt": "facebook/opt-350m",
        "bert_large": "bert-large-uncased",
        "gpt2_large": "gpt2-large",
    }
    model = AutoModelForSequenceClassification.from_pretrained(f"checkpoints/{task}_{model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_dict[model_name])
    if 'gpt2' in model_name:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        padding_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
        model.config.pad_token_id = padding_token_id
        tokenizer.pad_token_id = padding_token_id
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model_name', type=str, default='bert')
    parser.add_argument('-t','--task', type=str, default='sst')
    args = parser.parse_args()
    model_name = args.model_name
    task = args.task

    model, tokenizer = load_model(model_name, task)

    test(model, tokenizer, task)

if __name__ == "__main__":
    main()