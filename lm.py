import pandas as pd
import argparse 
import os
from saliency import *
from tqdm import tqdm

def load_model(model_name="gpt2", tuned=False, task="zuco11"):
    task_dict = {"zuco11": "sst", "zuco13": "wiki"}
    num_dict = {"zuco11": 3, "zuco13": 9}
    model_dict = {
        "bert": "bert-base-uncased", 
        "roberta": "roberta-base", 
        "gpt2": "gpt2", 
        "distilbert": "distilbert-base-uncased"
    }
    if model_name == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained(model_dict[model_name])
        if tuned:
            model = GPT2ForSequenceClassification.from_pretrained(f'checkpoints/{task_dict[task]}_gpt2', num_labels=num_dict[task])
        else:
            # load random init model
            model = GPT2ForSequenceClassification.from_pretrained(model_dict[model_name], num_labels=num_dict[task])
            model.init_weights()
    elif model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if tuned:
            model = BertForSequenceClassification.from_pretrained(f'checkpoints/{task_dict[task]}_bert', num_labels=num_dict[task])
        else:
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_dict[task])
            model.init_weights()
    elif model_name == "roberta":
        tokenizer = AutoTokenizer.from_pretrained(model_dict[model_name])
        if tuned:
            model = RobertaForSequenceClassification.from_pretrained(f'checkpoints/{task_dict[task]}_roberta', num_labels=num_dict[task])
        else:
            model = RobertaForSequenceClassification.from_pretrained(model_dict[model_name], num_labels=num_dict[task])
            
    elif model_name == "distilbert":
        tokenizer = AutoTokenizer.from_pretrained(model_dict[model_name])
        if tuned:
            model = DistilBertForSequenceClassification.from_pretrained(f'checkpoints/{task_dict[task]}_distilbert', num_labels=num_dict[task])
        else:
            model = DistilBertForSequenceClassification.from_pretrained(model_dict[model_name], num_labels=num_dict[task])
    else:
        raise ValueError("Invalid model name")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    return tokenizer, model

def load_data(task):
    if task == "zuco11":
        test_df = pd.read_csv("data/sst/test.csv", sep=",")
    elif task == "zuco13":
        test_df = pd.read_csv("data/wiki/test_.csv", sep=",")
    else:
        raise ValueError("Invalid task name")

    return test_df

def seq_saliency(text, label, tokenizer, model, task):
    if task == "zuco11":
        label_mapping = {0: 0, 2: 1, 4: 2}
    elif task == "zuco13":
        label_mapping = {'award': 0, 'education':1, 'employer':2, 'founder':3, 
                        'job_title':4, 'nationality':5, 'political_affiliation':6, 'visited':7, 'wife':8}

    label_id = label_mapping[label]
    
    input_text = text.strip()
    inputs = tokenizer(input_text, return_tensors="pt")

    input_tokens = inputs["input_ids"].squeeze().tolist()
    attention_ids = inputs["attention_mask"].squeeze().tolist()

    tokens, saliency_matrix, embd_matrix = lm_saliency(model, tokenizer, input_tokens, attention_ids, label_id)
    x_grad = input_x_gradient(tokens, text, saliency_matrix, embd_matrix, model, normalize=True)
    l1_grad = l1_grad_norm(tokens, text, saliency_matrix, model, normalize=True)
    l2_grad = l2_grad_norm(tokens, text, saliency_matrix, model, normalize=True)
    return x_grad, l1_grad, l2_grad

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--task', type=str, default='zuco12')
    parser.add_argument('-m','--model_name', type=str, default='gpt2')
    parser.add_argument('--tuned', action='store_true', help='finetuned model')
    args = parser.parse_args()
    model_name = args.model_name
    tuned = args.tuned
    task = args.task
    
    tokenizer, model = load_model(model_name, tuned=tuned, task=task)
    num_labels = model.config.num_labels    

    data = load_data(task)
    task_dict = {"zuco11": "task1", "zuco13": "task3"}
    df_saliency = pd.DataFrame(columns=["sid", "x_grad", "l1_grad", "l2_grad"])
    
    for i in tqdm(range(len(data))):
        x_grad, l1_grad, l2_grad = seq_saliency(data.iloc[i]["text"], data.iloc[i]["label"], tokenizer, model, task)
        new_row = pd.DataFrame({
            "sid": [data.iloc[i]["sid"]],
            "x_grad": [x_grad.tolist()],
            "l1_grad": [l1_grad.tolist()],
            "l2_grad": [l2_grad.tolist()]
        })
        df_saliency = pd.concat([df_saliency, new_row], ignore_index=True)
    if tuned:
        output_path = f"data/zuco/{task_dict[task]}/{model_name}_saliency.csv"
    else:
        output_path = f"data/zuco/{task_dict[task]}/{model_name}_control_saliency.csv"

    df_saliency.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
