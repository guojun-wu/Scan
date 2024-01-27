import pandas as pd
import numpy as np
import string
import ast
import scipy.stats 
import argparse 

def read_bnc():
    with open('data/bnc.txt', 'r', encoding='latin-1') as file:
        content = file.read()

    lines = [line.strip().split('\t') for line in content.split('\n') if line.strip()]
    headers = lines[0]
    bnc_df = pd.DataFrame(lines[1:], columns=headers)
    bnc_df['Freq'] = bnc_df['Freq'].fillna(0)
    bnc_df['Freq'] = bnc_df['Freq'].astype(int)
    bnc_df['Word'] = bnc_df['Word'].str.lower()
    bnc_df['Word'] = bnc_df['Word'].str.replace(' ', '')

    return bnc_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--task', type=str, default='sst')
    args = parser.parse_args()

    task = args.task

    bnc_df = read_bnc()
    test_df = pd.read_csv(f'data/{task}/test.csv', sep=',')
    test_df['text'] = test_df['text'].str.lower()
    freq_df = pd.DataFrame([], columns=['sid', 'freq'])
    
    for i in range(len(test_df)):
        freq_list = []
        for word in test_df['text'][i].split():
            word = word.translate(str.maketrans('', '', string.punctuation))
            if word in bnc_df['Word'].values:
                freq = int(bnc_df[bnc_df['Word'] == word]['Freq'].values[0])
                if freq == 0:
                    freq = 10
                inverse_freq = -np.log(freq/1000000)
                freq_list.append(inverse_freq)
            else:
                freq = 10
                inverse_freq = -np.log(freq/1000000)
                freq_list.append(inverse_freq)
        
        # add the frequency list to the dataframe
        freq_df = pd.concat([freq_df, pd.DataFrame({'sid': [test_df['sid'][i]], 'freq': [freq_list]})], ignore_index=True)

    # normalize the frequencies in each sentence
    freq_df['freq'] = freq_df['freq'].apply(lambda x: x/np.sum(x))
    freq_df['freq'] = freq_df['freq'].apply(lambda x: x.tolist())
    # make the frequency list a string
    freq_df.to_csv(f'data/{task}/freq.csv', index=False)

if __name__ == "__main__":
    main()
