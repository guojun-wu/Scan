import pandas as pd
import numpy as np
import os
import ast
import argparse 
import scipy.stats  
import matplotlib.pyplot as plt
from CONTANTS import subj_sst_acc, subj_wiki_acc, task_title, RESULT_PATH
from matplotlib.font_manager import FontProperties


name_dict = {"bert": "BERT_BASE", "bert_large": "BERT_Large", "roberta": "RoBERTa", "distilbert": "DistilBERT", "gpt2": "GPT2", "gpt2_large": "GPT2_Large", "opt": "OPT"}
def read_data(tuned, task):
    models = ["bert", "bert_large", "roberta", "distilbert", "gpt2", "gpt2_large", "opt"]
    fix_df = pd.read_csv(f"{RESULT_PATH}/{task}/fixation.csv", sep=",")
    df = fix_df.rename(columns={"list_dur": "fixation"})
    for model in models:
        model_df = pd.read_csv(f"{RESULT_PATH}/{task}/{model}_{tuned}_saliency.csv", sep=",")
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

def subject_analysis(task, model_name):
    corr_df = pd.DataFrame()
    subj_df = pd.read_csv(f'{RESULT_PATH}/{task}/fixation_subj.csv', sep=",")
    model_df = pd.read_csv(f'{RESULT_PATH}/{task}/{model_name}_finetuned_saliency.csv', sep=",")
    subj_df['list_dur'] = subj_df['list_dur'].apply(ast.literal_eval)
    model_df['l1_grad'] = model_df['l1_grad'].apply(ast.literal_eval)
    subjs = subj_df['id'].unique()
    for subj in subjs:
        df = subj_df[subj_df['id'] == subj]
        df = df.merge(model_df, on="sid")
        
        corr_list = []
        for i in range(len(df)):
            if len(df["list_dur"][i]) != len(df["l1_grad"][i]):
                raise ValueError(f"Length of {col} and fixation is not the same for {i}")
            corr, _ = scipy.stats.spearmanr(df["l1_grad"][i], df["list_dur"][i])
            corr_list.append(corr)
        # pad the list to the same length
        corr_list += [np.nan] * (len(model_df) - len(corr_list))
        corr_df[subj] = corr_list
    corr_df.to_csv(f'data/{task}/{model_name}_subj_corr.csv', index=False)    

def draw_subject_barplot(task_names=["sst", "wiki"], model_name="gpt2"):
    fig, axs = plt.subplots(1, len(task_names), figsize=(14, 4), sharey=True)

    for j, task in enumerate(task_names):
        mean_df = pd.read_csv(f'{RESULT_PATH}/{task}/{model_name}_subj_corr.csv', sep=",")
        mean = mean_df.mean()
        std_error = mean_df.std() / np.sqrt(len(mean_df))
        
        if task == "sst":
            mean = mean.sort_index(key=lambda x: [subj_sst_acc[subj] for subj in x])
            std_error = std_error.sort_index(key=lambda x: [subj_sst_acc[subj] for subj in x])
        else:
            mean = mean.sort_index(key=lambda x: [subj_wiki_acc[subj] for subj in x])
            std_error = std_error.sort_index(key=lambda x: [subj_wiki_acc[subj] for subj in x])

        bars = axs[j].bar(np.arange(len(mean)), mean, yerr=std_error, capsize=5,
                          color='skyblue', edgecolor='black', label=task.upper())

        for bar, subj in zip(bars, mean.index):
            acc = subj_sst_acc[subj] if task == "sst" else subj_wiki_acc[subj]
            axs[j].text(bar.get_x() + bar.get_width() / 2, 0.09,
                        f'{subj}', ha='center', va='center', color='black', fontsize=11)
            axs[j].text(bar.get_x() + bar.get_width() / 2, 0.11,
                        f'{acc:.2f}', ha='center', va='center', color='black', fontsize=8, style='italic')

        axs[j].set_title(task_title[task], fontsize=12)
        axs[j].set_xticks([])

        axs[j].set_ylim(0.1, 0.45)
        if j == 0:
            axs[j].set_ylabel('Spearman Correlation', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{model_name}_subject_correlation.png', dpi=300)

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
        model_names = ["bert", "gpt2"]
        for model_name in model_names:
            subject_analysis(task, model_name)
        draw_subject_barplot()
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

