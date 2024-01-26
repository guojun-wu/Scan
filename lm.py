import pandas as pd
import argparse 
import os
from saliency import *
from tqdm import tqdm

def load_model(model_name="gpt2", tuned=False, task="sst"):
    num_dict = {"sst": 3, "wiki": 9}
    model_dict = {
        "bert": BertForSequenceClassification, 
        "roberta": RobertaForSequenceClassification, 
        "gpt2": GPT2ForSequenceClassification, 
        "distilbert": DistilBertForSequenceClassification,
        "opt": OPTForSequenceClassification}
    path_dict = {
        "bert": "bert-base-uncased", 
        "roberta": "roberta-base", 
        "gpt2": "gpt2", 
        "distilbert": "distilbert-base-uncased",
        "opt": "facebook/opt-350m",}

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path_dict[model_name])

    # Load model based on model_name
    if tuned:
        model = model_dict[model_name].from_pretrained(f'checkpoints/{task}_{model_name}', num_labels=num_dict[task])
    else:
        model = model_dict[model_name].from_pretrained(path_dict[model_name], num_labels=num_dict[task])
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model

def seq_saliency(text, label, tokenizer, model, task):
    if task == "sst":
        label_mapping = {0: 0, 2: 1, 4: 2}
    elif task == "wiki":
        label_mapping = {'award': 0, 'education':1, 'employer':2, 'founder':3, 
                        'job_title':4, 'nationality':5, 'political_affiliation':6, 'visited':7, 'wife':8}

    label_id = label_mapping[label]
    
    input_text = text.strip()
    inputs = tokenizer(input_text, return_tensors="pt")

    input_tokens = inputs["input_ids"].squeeze().tolist()
    attention_ids = inputs["attention_mask"].squeeze().tolist()

    tokens, saliency_matrix, embd_matrix = lm_saliency(model, tokenizer, input_tokens, attention_ids, label_id)
    # x_grad = input_x_gradient(tokens, text, saliency_matrix, embd_matrix, model, normalize=True)
    l1_grad = l1_grad_norm(tokens, text, saliency_matrix, model, normalize=True)
    # l2_grad = l2_grad_norm(tokens, text, saliency_matrix, model, normalize=True)
    return l1_grad

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--task', type=str, default='sst')
    parser.add_argument('-m','--model_name', type=str, default='gpt2')
    parser.add_argument('--tuned', action='store_true', help='finetuned model')
    args = parser.parse_args()
    model_name = args.model_name
    tuned = args.tuned
    task = args.task
    
    tokenizer, model = load_model(model_name, tuned=tuned, task=task)
    num_labels = model.config.num_labels    

    data = pd.read_csv(f"data/{task}/test.csv", sep=",")
    df_saliency = pd.DataFrame(columns=["sid", "l1_grad"])
    
    for i in tqdm(range(len(data))):
        l1_grad= seq_saliency(data.iloc[i]["text"], data.iloc[i]["label"], tokenizer, model, task)
        new_row = pd.DataFrame({
            "sid": [data.iloc[i]["sid"]],
            "l1_grad": [l1_grad.tolist()],
        })
        df_saliency = pd.concat([df_saliency, new_row], ignore_index=True)
    if tuned:
        output_path = f"data/{task}/{model_name}_saliency.csv"
    else:
        output_path = f"data/{task}/{model_name}_control_saliency.csv"

    df_saliency.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
