import pandas as pd

def load_train_data():
    data = pd.read_csv('../../data/pf_suvs_i302_1s2024.csv')
    return data

def load_eval_data(path):
    data = pd.read_csv(path)
    return data