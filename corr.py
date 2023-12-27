import pandas as pd
import numpy as np
import os
import ast
import scipy.stats

def average_fixation(path):
    # average the fixation data of all the subjects
    df = pd.read_csv(path, sep=',')
    df_avg = pd.DataFrame(columns=['sn', 'list_dur'])
    sentences = df['sn'].unique()
    for sentence in sentences:
        df_sentence = df[df['sn'] == sentence]
        df_tmp = df_sentence['list_dur'].apply(ast.literal_eval)
        avg_dur = np.mean(df_tmp.values.tolist(), axis=0)
        df_avg = pd.concat([df_avg, pd.DataFrame({'sn': sentence, 'list_dur': [avg_dur.tolist()]})], ignore_index=True)
    return df_avg

def avergae_saliency(path):
    df = pd.read_csv(path, sep=',')
    df_avg = pd.DataFrame(columns=['sn', 'x_grad', 'l1_grad', 'l2_grad'])
    sentences = df['sn'].unique()
    for sentence in sentences:
        df_sentence = df[df['sn'] == sentence]
        df_tmp = df_sentence['x_grad'].apply(ast.literal_eval)
        avg_x = np.mean(df_tmp.values.tolist(), axis=0)
        df_tmp = df_sentence['l1_grad'].apply(ast.literal_eval)
        avg_l1 = np.mean(df_tmp.values.tolist(), axis=0)
        df_tmp = df_sentence['l2_grad'].apply(ast.literal_eval)
        avg_l2 = np.mean(df_tmp.values.tolist(), axis=0)
        df_avg = pd.concat([df_avg, pd.DataFrame({'sn': sentence, 'x_grad': [avg_x.tolist()], 'l1_grad': [avg_l1.tolist()], 'l2_grad': [avg_l2.tolist()]})], ignore_index=True)
    return df_avg

def main():
    path = "data/zuco/task2/fixation.csv"
    df_fixation = average_fixation(path)
    path = "data/zuco/task2/bert_saliency.csv"
    df_saliency = avergae_saliency(path)
    # compute spearman correlation between fixation and saliency
    df = pd.merge(df_fixation, df_saliency, on="sn")
    # compute spearman correlation between fixation and x_explanation
    fixation = df['list_dur']
    x_explanation = df['x_grad']
    l1_explanation = df['l1_grad']
    l2_explanation = df['l2_grad']
    
    # compute spearman correlation between fixation and x_explanation/l1_explanation
    spearman_x = []
    spearman_l1 = []
    spearman_l2 = []

    for i in range(len(fixation)):
        if len(fixation[i]) != len(l1_explanation[i]):
            print(i)
            continue
        spearman_x.append(scipy.stats.spearmanr(fixation[i], x_explanation[i])[0])
        spearman_l1.append(scipy.stats.spearmanr(fixation[i], l1_explanation[i])[0])
        spearman_l2.append(scipy.stats.spearmanr(fixation[i], l2_explanation[i])[0])
    print(np.mean(spearman_x))
    print(np.mean(spearman_l1))
    print(np.mean(spearman_l2))
    

if __name__ == "__main__":
    main()
