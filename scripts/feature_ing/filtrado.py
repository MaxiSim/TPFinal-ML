import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random

def read_data(filename):
    data = pd.read_csv(filename)
    return data


def histogram_of_feature(feature):
    data = read_data('../../data/pf_suvs_i302_1s2024.csv')
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data[feature]))
    fig.update_layout(title_text=f'Histogram of feature "{feature}"', xaxis_title_text=feature, yaxis_title_text='Frequency')
    try:
        #checkeo si el feature es un numero
        check = float(data[feature][random.randint(0, len(data))])
        fig.update_xaxes(range=[data[feature].min(), data[feature].max()])  # Adjust x-axis range
    except:
        print(f'Feature "{feature}" is not numeric. The type is {type(data[feature][random.randint(0, len(data))])}')
        pass
    fig.show()

def ars_to_usd(data, price_usd):
    if data['oneda'] == "$":
        return data['monto'] / price_usd
    else:
        return data['monto']