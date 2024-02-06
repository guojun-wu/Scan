import pandas as pd
import numpy as np
import os
import ast
import argparse
from CONTANTS import DATA_PATH, RESULT_PATH

def average_fixation(df):
    # average the fixation data of all the subjects
    df_avg = pd.DataFrame(columns=['sid', 'list_dur'])
    sentences = df['sn'].unique()
    for sentence in sentences:
        df_sentence = df[df['sn'] == sentence]
        # get relative fixation duration
        df_tmp = df_sentence['list_dur'].apply(lambda x: [i/sum(x) for i in x])
        avg_dur = np.mean(df_tmp.values.tolist(), axis=0)
        df_avg = pd.concat([df_avg, pd.DataFrame({'sid': sentence, 'list_dur': [avg_dur.tolist()]})], ignore_index=True)
    return df_avg

def extract_fixation_data(path, task):
    # path is the path of the fixation data
    df_dur = pd.DataFrame(columns=['id', 'sid', 'list_dur'])
    df = pd.read_csv(path, sep='\t')
    subjects = df['id'].unique()
    for subject in subjects:
        df_subject = df[df['id'] == subject]
        df_subject = df_subject[['id', 'sn', 'nw', 'wn', 'dur']]  
        sentences = df_subject['sn'].unique()
        for sentence in sentences:
            df_sentence = df_subject[df_subject['sn'] == sentence]
            list_dur = np.zeros((df_sentence['nw'].max()), dtype=int)
            words = df_sentence['wn'].unique()
            
            for word in words:
                df_word = df_sentence[df_sentence['wn'] == word]
                duration = df_word['dur'].sum()
                list_dur[word-1] = duration
            df_dur = pd.concat([df_dur, pd.DataFrame({'id': subject, 'sid': sentence, 'list_dur': [list_dur.tolist()]})], ignore_index=True)
    df_dur.to_csv(f'{RESULT_PATH}/{task}/fixation_subj.csv', index=False)
    df_avg = average_fixation(df_dur)
    df_avg.to_csv(f'{RESULT_PATH}/{task}/fixation.csv', index=False)
                
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--task', type=str, default='task1', help='task1 or task3')
    args = parser.parse_args()

    task = args.task
    task_dict = {'task1': 'sst', 'task3': 'wiki'}

    if not os.path.exists(f"{RESULT_PATH}/{task}"):
        os.makedirs(f"{RESULT_PATH}/{task}")

    path = f"{DATA_PATH}/{task}/Matlab_files/scanpath.csv"
    extract_fixation_data(path, task_dict[task]) 

if __name__ == "__main__":
    main()