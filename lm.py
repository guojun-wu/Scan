import pandas as pd
import argparse 
import os
from saliency import *
from tqdm import tqdm

def load_model(model_name="gpt2", control=False):
    if model_name == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        if control:
            # random init
            config = GPT2Config()
            model = GPT2LMHeadModel(config)
        else:
            model = GPT2LMHeadModel.from_pretrained(model_name)
    elif model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if control:
            # random init
            config = BertConfig()
            model = BertForMaskedLM(config)
        else:
            model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    else:
        raise ValueError("Invalid model name")
    return tokenizer, model

def load_data(task):
    if task == "zuco11":
        base_path = "data/zuco/task1/Matlab_files"
    elif task == "zuco12":
        base_path = "data/zuco/task2/Matlab_files"
    elif task == "zuco13":
        base_path = "data/zuco/task3/Matlab_files"
    else:
        raise ValueError("Invalid task name")

    data = pd.DataFrame(columns=["id", "sn", "input", "output"])
    sentences = pd.read_csv(os.path.join(base_path, "sentence_content.csv"), sep="\t")
    scanpaths = pd.read_csv(os.path.join(base_path, "scanpath_content.csv"), sep="\t")
    # loop based on id and sn in scanpaths
    for i in range(len(scanpaths)):
        subject = scanpaths.iloc[i]["id"]
        sn = scanpaths.iloc[i]["SN"]
        input_seq = sentences[sentences["SN"] == sn]["CONTENT"].values[0]
        output_seq = scanpaths.iloc[i]["CONTENT"]
        data = pd.concat([data, pd.DataFrame({"id": subject, "sn": sn, "input": input_seq, "output": output_seq}, index=[0])], ignore_index=True)
    return data

def seq_saliency(input_seq, output_seq, tokenizer, model):
    
    output_tokens = tokenizer(output_seq)['input_ids']
    if isinstance(model, GPT2LMHeadModel):
        input_seq = input_seq.strip() + " " * len(output_tokens)
    elif isinstance(model, BertForMaskedLM):
        input_seq = input_seq.strip() + "[MASK]" * len(output_tokens)

    input_tokens = tokenizer(input_seq)['input_ids']
    attention_ids = tokenizer(input_seq)['attention_mask']

    tokens, saliency_matrix, embd_matrix = lm_saliency(model, tokenizer, input_tokens, attention_ids, output_tokens)
    x_grad = input_x_gradient(tokens, saliency_matrix, embd_matrix, model, normalize=True)
    l1_grad = l1_grad_norm(tokens, saliency_matrix, model, normalize=True)
    l2_grad = l2_grad_norm(tokens, saliency_matrix, model, normalize=True)
    return x_grad, l1_grad, l2_grad

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model_name', type=str, default='gpt2')
    parser.add_argument('--control', action='store_true', help='control mode')
    parser.add_argument('--test', action='store_true', help='test mode')
    args = parser.parse_args()
    model_name = args.model_name
    control = args.control
    
    tokenizer, model = load_model(model_name, control=control)

    data = load_data("zuco12")
    df_saliency = pd.DataFrame(columns=["id", "sn", "x_grad", "l1_grad", "l2_grad"])
    if args.test:
        data = data[:10]
    for i in tqdm(range(len(data))):
        x_grad, l1_grad, l2_grad = seq_saliency(data.iloc[i]["input"], 
                                    data.iloc[i]["output"], tokenizer, model)
        new_row = pd.DataFrame({
            "id": [data.iloc[i]["id"]],
            "sn": [data.iloc[i]["sn"]],
            "x_grad": [x_grad.tolist()],
            "l1_grad": [l1_grad.tolist()],
            "l2_grad": [l2_grad.tolist()]
        })
        df_saliency = pd.concat([df_saliency, new_row], ignore_index=True)
    if not control:
        output_path = f"data/zuco/task2/{model_name}_saliency.csv"
    else:
        output_path = f"data/zuco/task2/{model_name}_control_saliency.csv"
    df_saliency.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
