import numpy as np
import pandas as pd
from data_loader.load_func import load_train_data, load_eval_data


class Engineer:
    
    def __init__(self, df):
        self.data = df
        
    def add_features(self):
        self.data['new_feature'] = self.data['feature1'] + self.data['feature2']
        return self.data
    

