import pandas as pd
import numpy as np
import plotly.graph_objects as go

def read_data(filename):
    data = pd.read_csv(filename)
    return data