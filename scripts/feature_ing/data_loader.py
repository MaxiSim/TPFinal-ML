import pandas as pd

def read_data(filename):
    data = pd.read_csv(filename)
    return data