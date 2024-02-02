import pandas as pd
import numpy as np
import os
import ast
import argparse 
import scipy.stats  
import matplotlib.pyplot as plt

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
    # get the mean and standard error of correlations
    mean = {}
    std_error = {}
    for key in corr_df.keys():
        mean[name_dict[key]] = np.mean(corr_df[key])
        std_error[name_dict[key]] = np.std(corr_df[key])/np.sqrt(len(corr_df[key]))

    # Create pd dataframe for mean and std
    corr_df = pd.DataFrame([mean, std_error])
    corr_df = corr_df.rename(index={0: "Mean", 1: "Standard Error"})
    corr_df = corr_df.T
    corr_df = corr_df.round(3)

    # Save to tex
    corr_df.to_latex("correlation.tex")

    print(corr_df)


def draw_boxplot(corr_df):
    # get the mean and std of correlations
    mean = {}
    std = {}
    for key in corr_df.keys():
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--task", type=str, default="sst")
    parser.add_argument("--tuned", type=str, default="finetuned")
    args = parser.parse_args()

    tuned = args.tuned
    task = args.task

    if tuned == "random":
        # for every random seed, generate the correlation
        seeds = [str(i) for i in range(1, 2)]
        df = read_data(tuned, task)
        corr_df = get_corr(df)
        for seed in seeds:
            tmp_df = read_data(f"random{seed}", task)
            tmp_corr_df = get_corr(tmp_df)
            for key in corr_df.keys():
                corr_df[key] += tmp_corr_df[key]
        for key in corr_df.keys():
            corr_df[key] = [x/len(seeds) for x in corr_df[key]] 
    else:
        df = read_data(tuned, task)
        corr_df = get_corr(df)
        
    generate_tex(corr_df)
    # draw_boxplot(corr_df)


if __name__ == "__main__":
    main()

