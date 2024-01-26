import pandas as pd
import numpy as np
import os
import ast
import argparse 
import scipy.stats    

def average_fixation(path):
    # average the fixation data of all the subjects
    df = pd.read_csv(path, sep=',')
    df_avg = pd.DataFrame(columns=['sid', 'list_dur'])
    sentences = df['sn'].unique()
    for sentence in sentences:
        df_sentence = df[df['sn'] == sentence]
        df_tmp = df_sentence['list_dur'].apply(ast.literal_eval)
        # normalize the fixation duration
        df_tmp = df_tmp.apply(lambda x: [i/sum(x) for i in x])
        avg_dur = np.mean(df_tmp.values.tolist(), axis=0)
        df_avg = pd.concat([df_avg, pd.DataFrame({'sid': sentence, 'list_dur': [avg_dur.tolist()]})], ignore_index=True)
    return df_avg
    
def compute_spearman(df):
    # compute spearman correlation between fixation and x_explanation
    print(len(df))
    fixation = df['list_dur']
    x_explanation = df['x_grad'].apply(ast.literal_eval)
    l1_explanation = df['l1_grad'].apply(ast.literal_eval)
    l2_explanation = df['l2_grad'].apply(ast.literal_eval)

    spearman_x = []
    spearman_l1 = []
    spearman_l2 = []
    
    for i in range(len(fixation)):
        if len(fixation[i]) != len(l1_explanation[i]):
            print("Error: length of fixation and explanation does not match")
            break
        spearman_x.append(scipy.stats.spearmanr(fixation[i], x_explanation[i])[0])
        spearman_l1.append(scipy.stats.spearmanr(fixation[i], l1_explanation[i])[0])
        spearman_l2.append(scipy.stats.spearmanr(fixation[i], l2_explanation[i])[0])
    print("x_explanation")
    print(np.mean(spearman_x))
    print("l1_explanation")
    print(np.mean(spearman_l1))
    print("l2_explanation")
    print(np.mean(spearman_l2))

    
    # concat the vectors in fixation 
    # fixation_concat = []
    # for i in range(len(fixation)):
    #     fixation_concat += fixation[i]
    # x_explanation_concat = []
    # for i in range(len(x_explanation)):
    #     x_explanation_concat += x_explanation[i]
    # l1_explanation_concat = []
    # for i in range(len(l1_explanation)):
    #     l1_explanation_concat += l1_explanation[i]
    # l2_explanation_concat = []
    # for i in range(len(l2_explanation)):
    #     l2_explanation_concat += l2_explanation[i]

    # # compute spearman correlation
    # print("x_explanation")
    # print(scipy.stats.spearmanr(fixation_concat, x_explanation_concat))
    # print("l1_explanation")
    # print(scipy.stats.spearmanr(fixation_concat, l1_explanation_concat))
    # print("l2_explanation")
    # print(scipy.stats.spearmanr(fixation_concat, l2_explanation_concat))
   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--task', type=str, default='zuco12')
    parser.add_argument('-m','--model_name', type=str, default='gpt2')
    parser.add_argument('--tuned', action='store_true', help='finetuned model')
    args = parser.parse_args()

    task_dict = {"zuco11": "task1", "zuco13": "task3"}
    base_path = "data/zuco/" + task_dict[args.task]

    model_name = args.model_name
    
    fixation_path = os.path.join(base_path, "fixation.csv")
    if args.tuned:
        saliency_path = os.path.join(base_path, args.model_name + "_saliency.csv")
        print("finetuned model")
    else:
        saliency_path = os.path.join(base_path, args.model_name + "_control_saliency.csv")
        
    df_fixation = average_fixation(fixation_path)
    df_saliency = pd.read_csv(saliency_path, sep=',')
    df = pd.merge(df_fixation, df_saliency, on='sid', how='inner')
    # compute spearman correlation between fixation and saliency
    compute_spearman(df)
    

if __name__ == "__main__":
    main()
