import pandas as pd
import numpy as np
import os
import ast

def main():
    freq_df = pd.read_csv('data/1_2_all_freq.txt', sep='\t')