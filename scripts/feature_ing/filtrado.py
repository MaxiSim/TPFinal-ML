import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random

def read_data(filename):
    data = pd.read_csv(filename)
    return data

def histogram_of_feature(feature, path_data):
    data = read_data(path_data)
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
    data.loc[data['Moneda'] == "$", 'Precio'] = round(data.loc[data['Moneda'] == "$", 'Precio'] / price_usd, 2)
    data.loc[data['Moneda'] == "$", 'Moneda'] = "U$S"
    return data

def rewrite_data(data, filename):
    data.to_csv(filename, index=False)

def delete_rows(filename, index, newfile=None):
    data = read_data(filename)
    data = data.drop(index)
    if newfile == None:
        rewrite_data(data, filename)
    else:
        rewrite_data(data, newfile)
        
def delete_columns(filename, feature, newfile=None):
    data = read_data(filename)
    data = data.drop(feature, axis=1)
    if newfile == None:
        rewrite_data(data, filename)
    else:
        rewrite_data(data, newfile)
    
def find_outliers(data, feature, threshold):
    num_outliers = []
    for i in range(len(threshold)):
           outliers = data.loc[data[feature] >= threshold[i]] 
           num_outliers.append(outliers.shape[0])
    return num_outliers, threshold

def plot (outliers, thresholds):
    fig = go.Figure()
    # fig.add_trace(go.Box(y=outliers, boxpoints='outliers', jitter=0.3, pointpos=-1.8))  # Mostrar outliers
    fig.add_trace(go.Scatter(x=thresholds, y=outliers, mode='markers', marker=dict(color='red', size=6), showlegend=False))  # Mostrar umbrales

    # Personalizar el diseño del gráfico
    fig.update_layout(title='Gráfico de Box con Outliers y Thresholds',
                    xaxis=dict(title='Thresholds'),
                    yaxis=dict(title='Valores'),
                    showlegend=False)

    # Mostrar el gráfico
    fig.show()