import pandas as pd
import os
from saliency import *

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def load_data(task):
    if task == "zuco11":
        base_path = "data/zuco/task1/Matlab_files"
    elif task == "zuco12":
        base_path = "data/zuco/task2/Matlab_files"
    elif task == "zuco13":
        base_path = "data/zuco/task3/Matlab_files"
    else:
        raise ValueError("Invalid task name")

    # load data
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

def seq_saliency(input_seq, output_seq):
    # tokenize output sequences
    output_tokens = tokenizer(output_seq)['input_ids']

    # add whitespace to input_seq for tokens in output_seq
    input_seq = input_seq.strip() + " " * (len(output_tokens))

    # tokenize input sequence
    input_tokens = tokenizer(input_seq)['input_ids']
    attention_ids = tokenizer(input_seq)['attention_mask']

    saliency_matrix, embd_matrix = lm_saliency(model, input_tokens, attention_ids, output_tokens)
    x_explanation = input_x_gradient(saliency_matrix, embd_matrix, normalize=True)
    l1_explanation = l1_grad_norm(saliency_matrix, normalize=True)
    return x_explanation, l1_explanation

def main():
    data = load_data("zuco12")
    df_saliency = pd.DataFrame(columns=["id", "sn", "x_explanation", "l1_explanation"])
    for i in range(len(data)):
        x_explanation, l1_explanation = seq_saliency(data.iloc[i]["input"], data.iloc[i]["output"])
        # x_explanation and l1_explanation are numpy arrays
        new_row = pd.DataFrame({
            "id": [data.iloc[i]["id"]],
            "sn": [data.iloc[i]["sn"]],
            "x_explanation": [x_explanation.tolist()],
            "l1_explanation": [l1_explanation.tolist()]
        })
        df_saliency = pd.concat([df_saliency, new_row], ignore_index=True)
    df_saliency.to_csv("data/zuco/task2/saliency.csv", index=False)

if __name__ == "__main__":
    main()
