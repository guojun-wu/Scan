import pandas as pd
import numpy as np
import os
import ast
import argparse 
import scipy.stats    

def average_fixation(path):
    # average the fixation data of all the subjects
    df = pd.read_csv(path, sep=',')
    df_avg = pd.DataFrame(columns=['sn', 'list_dur'])
    sentences = df['sn'].unique()
    for sentence in sentences:
        df_sentence = df[df['sn'] == sentence]
        df_tmp = df_sentence['list_dur'].apply(ast.literal_eval)
        # normalize the fixation duration
        df_tmp = df_tmp.apply(lambda x: [i/sum(x) for i in x])
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

def compute_spearman(df_fixation, df_saliency, avg=False):
    if avg:
        df = pd.merge(df_fixation, df_saliency, on="sn")
    else:
        df = pd.merge(df_fixation, df_saliency, on=["id", "sn"])
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
    if avg:
        print("average spearman correlation between fixation and x_explanation/l1_explanation/l2_explanation")
    else:
        print("spearman correlation between fixation and x_explanation/l1_explanation/l2_explanation for each subject")
    print(np.mean(spearman_x))
    print(np.mean(spearman_l1))
    print(np.mean(spearman_l2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--task', type=str, default='zuco12')
    parser.add_argument('-m','--model_name', type=str, default='gpt2')
    parser.add_argument('--control', action='store_true', help='control mode')
    args = parser.parse_args()

    task_dict = {"zuco11": "task1", "zuco12": "task2"}
    base_path = "data/zuco/" + task_dict[args.task]

    model_name = args.model_name
    
    fixation_path = os.path.join(base_path, "fixation.csv")
    if args.control:
        saliency_path = os.path.join(base_path, args.model_name + "_control_saliency.csv")
        print("control mode")
    else:
        saliency_path = os.path.join(base_path, args.model_name + "_saliency.csv")
    df_fixation = average_fixation(fixation_path)
    df_saliency = avergae_saliency(saliency_path)
    # compute spearman correlation between fixation and saliency
    compute_spearman(df_fixation, df_saliency, avg=True)

    # compute spearman correlation between fixation and saliency for each subject
    df_fixation = pd.read_csv(fixation_path, sep=',')
    df_fixation['list_dur'] = df_fixation['list_dur'].apply(ast.literal_eval)
    df_saliency = pd.read_csv(saliency_path, sep=',')
    df_saliency['x_grad'] = df_saliency['x_grad'].apply(ast.literal_eval)
    df_saliency['l1_grad'] = df_saliency['l1_grad'].apply(ast.literal_eval)
    df_saliency['l2_grad'] = df_saliency['l2_grad'].apply(ast.literal_eval)
    compute_spearman(df_fixation, df_saliency, avg=False)


    
    

if __name__ == "__main__":
    main()
