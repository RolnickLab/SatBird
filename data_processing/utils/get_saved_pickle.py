import pandas

def save_pickle(df, path):
    """Save dataframe as a pickle"""
    df.to_pickle(path)