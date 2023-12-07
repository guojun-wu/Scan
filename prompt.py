# produce prompt with data/zuco/task1/Matlab_files/scanpath_content.csv and data/zuco/task1/Matlab_files/sentence_content.csv

import os
import pandas as pd
import numpy as np

def format_prompt(current_sn, current_id, scanpath_content, sentence_content):
    selected_id = get_ids(scanpath_content, current_id)
    prompt = ""
    for sid in selected_id:
        sn = np.random.choice(scanpath_content.loc[scanpath_content['id'] == sid]['SN'])
        while sn == current_sn:
            sn = np.random.choice(scanpath_content.loc[scanpath_content['id'] == sid]['SN'])
            
        sentence = sentence_content.loc[(sentence_content['SN'] == sn)]['CONTENT'].values[0]
        scanpath = scanpath_content.loc[(scanpath_content['id'] == sid) & (scanpath_content['SN'] == sn)]['CONTENT'].values[0]
        prompt += f"sentence: {sentence} scanpath: {scanpath} "
    current_sentence = sentence_content.loc[(sentence_content['SN'] == current_sn)]['CONTENT'].values[0]
    prompt += f"sentence: {current_sentence} scanpath: "
    
    return prompt

def get_ids(scanpath_content, current_id):
    # randomly select three ids that are not the same as current_id
    selected_id = [current_id]
    while len(selected_id) < 4:
        selected_id.append(np.random.choice(scanpath_content['id']))
        selected_id = list(set(selected_id))
    selected_id.remove(current_id)
    return selected_id  

def main():
    base_path = "data/zuco/task1/Matlab_files/"
    scanpath_content = pd.read_csv(os.path.join(base_path, "scanpath_content.csv"), sep='\t')
    sentence_content = pd.read_csv(os.path.join(base_path, "sentence_content.csv"), sep='\t')

    prompt_df = scanpath_content[['id', 'SN']]
    prompt_df['prompt'] = np.nan

    # random seed
    np.random.seed(40)

    for index, row in prompt_df.iterrows():
        prompt = format_prompt(row['SN'], row['id'], scanpath_content, sentence_content)
        prompt_df.at[index, 'prompt'] = prompt

    prompt_df.to_csv(os.path.join(base_path, "prompt.csv"), index=False)

if __name__ == "__main__":
    main()