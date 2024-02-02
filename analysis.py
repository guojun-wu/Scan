import pandas as pd
import numpy as np
import os
import ast
import argparse 
import scipy.stats  
import matplotlib.pyplot as plt
from config import subj_sst_acc, subj_wiki_acc

name_dict = {"bnc": "BNC", "bert": "BERT_BASE", "bert_large": "BERT_Large", "roberta": "RoBERTa", "distilbert": "DistilBERT", "gpt2": "GPT2", "gpt2_large": "GPT2_Large", "opt": "OPT"}
def read_data(tuned, task):
    models = ["bert", "bert_large", "roberta", "distilbert", "gpt2", "gpt2_large", "opt"]
    fix_df = pd.read_csv(f"data/{task}/fixation.csv", sep=",")
    if tuned == "finetuned":
        freq_df = pd.read_csv(f"data/{task}/freq.csv", sep=",")

        df = fix_df.merge(freq_df, on="sid")
        df = df.rename(columns={"freq": "bnc", "list_dur": "fixation"})
    else:
        df = fix_df
        df = df.rename(columns={"list_dur": "fixation"})
    for model in models:
        model_df = pd.read_csv(f"data/{task}/{model}_{tuned}_saliency.csv", sep=",")
        model_df = model_df[["sid", "l1_grad"]]
        model_df = model_df.rename(columns={"l1_grad": model})
        df = df.merge(model_df, on="sid")
    for col in df.columns:
        if col != "sid":
            df[col] = df[col].apply(ast.literal_eval)

    return df

def get_corr(df):
    corr_df = pd.DataFrame()
    for col in df.columns:
        corr_list = []
        if col not in ["sid", "fixation"]:
            for i in range(len(df)):
                if len(df[col][i]) != len(df["fixation"][i]):
                    raise ValueError(f"Length of {col} and fixation is not the same for {i}")
                corr, _ = scipy.stats.spearmanr(df[col][i], df["fixation"][i])
                corr_list.append(corr)
            corr_df[col] = corr_list
    return corr_df

def generate_tex(corr_df):
    # get the mean and standard error of correlations
    mean = {}
    std_error = {}
    for key in corr_df.columns:
        mean[name_dict[key]] = corr_df[key].mean()
        std_error[name_dict[key]] = np.std(corr_df[key])/np.sqrt(len(corr_df[key]))

    # Create pd dataframe for mean and std
    corr_df = pd.DataFrame([mean, std_error])
    corr_df = corr_df.rename(index={0: "Mean", 1: "Standard Error"})
    corr_df = corr_df.T
    corr_df = corr_df.round(3)

    print(corr_df)


def draw_boxplot(corr_df):
    # get the mean and std of correlations
    mean = {}
    std = {}
    for key in corr_df.columns:
        mean[name_dict[key]] = np.mean(corr_df[key])
        std[name_dict[key]] = np.std(corr_df[key])

    means = [mean[name] for name in name_dict.values()]
    stds = [std[name] for name in name_dict.values()]

    # Create boxplot
    fig, ax = plt.subplots()
    ax.bar(mean.keys(), means, yerr=stds, capsize=10)
    ax.set_ylabel('Correlation Coefficient')
    ax.set_title('Correlation Coefficients Boxplot')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save or display the plot
    plt.savefig('correlation_boxplot.png')
    plt.show()

def subject_analysis(task):
    corr_df = pd.DataFrame()
    subj_df = pd.read_csv(f'data/{task}/fixation_subj.csv', sep=",")
    bert_df = pd.read_csv(f'data/{task}/bert_finetuned_saliency.csv', sep=",")
    subj_df['list_dur'] = subj_df['list_dur'].apply(ast.literal_eval)
    bert_df['l1_grad'] = bert_df['l1_grad'].apply(ast.literal_eval)
    print(subj_df.head())
    subjs = subj_df['id'].unique()
    for subj in subjs:
        df = subj_df[subj_df['id'] == subj]
        df = df.merge(bert_df, on="sid")
        
        corr_list = []
        for i in range(len(df)):
            if len(df["list_dur"][i]) != len(df["l1_grad"][i]):
                raise ValueError(f"Length of {col} and fixation is not the same for {i}")
            corr, _ = scipy.stats.spearmanr(df["l1_grad"][i], df["list_dur"][i])
            corr_list.append(corr)
        # pad the list to the same length
        corr_list += [np.nan] * (len(bert_df) - len(corr_list))
        corr_df[subj] = corr_list
    print(corr_df.head())
    corr_df.to_csv(f'data/{task}/subj_corr.csv', index=False)    

def draw_subject_boxplot(task):
    subj_df = pd.read_csv(f'data/{task}/subj_corr.csv', sep=",")
    mean = subj_df.mean()
    std = subj_df.std() / np.sqrt(len(subj_df))
    print(mean)

    # sort the mean by the accuracy
    if task == "sst":
        mean = mean.sort_index(key=lambda x: [subj_sst_acc[subj] for subj in x])
    else:
        mean = mean.sort_index(key=lambda x: [subj_wiki_acc[subj] for subj in x])
    
    # Create dotplot
    fig, ax = plt.subplots()
    ax.bar(mean.index, mean, yerr=std, capsize=10)
    ax.set_title('Relation Extraction Subject Correlation')
    ax.set_ylabel('Spearman Correlation')
    ax.set_xlabel('Participants (ranked by accuracy from left to right)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save or display the plot
    plt.savefig(f'{task}_subject_correlation.png')
    plt.show()
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--task", type=str, default="sst")
    parser.add_argument("--tuned", type=str, default="finetuned")
    parser.add_argument("--subject", action="store_true")
    args = parser.parse_args()

    tuned = args.tuned
    task = args.task
    subject = args.subject

    if subject:
        subject_analysis(task)
        draw_subject_boxplot(task)
        return

    if tuned == "random":
        # for every random seed, generate the correlation
        seeds = [str(i) for i in range(1, 3)]
        print(seeds)
        df = read_data(tuned, task)
        corr_df = get_corr(df)
        for seed in seeds:
            tmp_df = read_data(f"random{seed}", task)
            tmp_corr_df = get_corr(tmp_df)
            for key in corr_df.keys():
                corr_df[key] += tmp_corr_df[key]
        for key in corr_df.keys():
            corr_df[key] = [x/(len(seeds)+1) for x in corr_df[key]] 
    else:
        df = read_data(tuned, task)
        corr_df = get_corr(df)

    generate_tex(corr_df)
    # draw_boxplot(corr_df)


if __name__ == "__main__":
    main()

