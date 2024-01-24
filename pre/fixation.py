import pandas as pd
import numpy as np
import os
import ast

def extract_fixation_data(path):
    # path is the path of the fixation data
    df_dur = pd.DataFrame(columns=['id', 'sn', 'list_dur'])
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
            df_dur = pd.concat([df_dur, pd.DataFrame({'id': subject, 'sn': sentence, 'list_dur': [list_dur.tolist()]})], ignore_index=True)
    df_dur.to_csv("data/zuco/task1/fixation.csv", index=False)

                
def main():

    path = "data/zuco/task1/Matlab_files/scanpath.csv"
    extract_fixation_data(path)
    

if __name__ == "__main__":
    main()

    
