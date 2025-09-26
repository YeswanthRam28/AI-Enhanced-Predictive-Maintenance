# 01_data_exploration.py

import pandas as pd
import os

def load_data(file_path):
    """
    Load a raw data file as a pandas DataFrame.
    """
    df = pd.read_csv(file_path, sep=' ', header=None, engine='python')
    df.dropna(axis=1, how='all', inplace=True)  # remove empty columns
    return df

def explore_data(df, head_rows=5):
    """
    Perform basic exploration: shape, head, and summary.
    """
    print("Data Shape:", df.shape)
    print("First few rows:\n", df.head(head_rows))
    print("Basic Stats:\n", df.describe())

if __name__ == "__main__":
    # Example usage
    raw_file = os.path.join('data', 'raw', 'train_FD001.txt')
    df = load_data(raw_file)
    explore_data(df)
