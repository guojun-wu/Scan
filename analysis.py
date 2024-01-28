import pandas as pd
import numpy as np
import os
import ast
import argparse 
import scipy.stats  

def read_data(task):
    models = ["bert", "bert_large", "roberta", "distilbert", "gpt2", "gpt2_large", "opt"]
    fix_df = pd.read_csv(f"data/{task}/fixation.csv", sep=",")
    freq_df = pd.read_csv(f"data/{task}/freq.csv", sep=",")

    df = fix_df.merge(freq_df, on="sid")
    df = df.rename(columns={"freq": "bnc", "list_dur": "fixation"})
    for model in models:
        model_df = pd.read_csv(f"data/{task}/{model}_saliency.csv", sep=",")
        model_df = model_df[["sid", "l1_grad"]]
        model_df = model_df.rename(columns={"l1_grad": model})
        df = df.merge(model_df, on="sid")
    for col in df.columns:
        if col != "sid":
            df[col] = df[col].apply(ast.literal_eval)

    return df

def get_corr(df):
    corr_df = {}
    for col in df.columns:
        if col not in ["sid", "fixation"]:
            corr_df[col] = []
            for i in range(len(df)):
                if len(df[col][i]) != len(df["fixation"][i]):
                    print("Error: length of fixation and explanation does not match")
                    print(col, i)
                    break
                corr_df[col].append(scipy.stats.spearmanr(df[col][i], df["fixation"][i])[0])
                corr_df[col] = [x for x in corr_df[col] if str(x) != 'nan']
    return corr_df

def generate_tex(corr_df):
    df = pd.DataFrame()

    for key in corr_df.keys():
        df[key] = [np.mean(corr_df[key])]
    # generate latex table
    print(df)
   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--task", type=str, default="sst")
    args = parser.parse_args()

    df = read_data(args.task)
    corr_df = get_corr(df)
    generate_tex(corr_df)


if __name__ == "__main__":
    main()

