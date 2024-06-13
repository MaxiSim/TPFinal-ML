import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random
import os
import time

# --------- Read and rewrite dataset ----------

def read_data(filename):
    data = pd.read_csv(filename)
    return data

def rewrite_data(data, filename):
    data.to_csv(filename, index=False)

def delete_rows(data, indexes, filename):
    indexes = indexes.sort_values(ascending=False)
    for index in indexes:
        data = data.drop(index)
    rewrite_data(data, filename)
        
def delete_columns(data, feature, filename):
    data = data.drop(feature, axis=1)
    rewrite_data(data, filename)

# --------- Plots and histograms ----------

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

# --------- Clean price column ----------

def ars_to_usd(data, price_usd):
    data.loc[data['Moneda'] == "$", 'Precio'] = round(data.loc[data['Moneda'] == "$", 'Precio'] / price_usd, 2)
    data.loc[data['Moneda'] == "$", 'Moneda'] = "U$S"
    return data


# --------- Clean km column ---------- 
        
def km_to_int (data, filename):
    try:
        data['Kilómetros'] = data['Kilómetros'].str.replace(' km', '').astype(int)
    except:
        print("The column 'Kilómetros' is already clean")
    rewrite_data(data, filename)
    
def find_km_outliers(data, threshold):
    def is_same_digit(num):
        if num == 0:
            return False
        num_str = str(num)
        return all((ch == num_str[0]) for ch in num_str)
    same_digit_outliers = data.loc[data['Kilómetros'].apply(is_same_digit)].index
    outliers = data.loc[data['Kilómetros'] >= threshold].index
    return outliers, same_digit_outliers

def avg_km_year(data):
    current_year = time.localtime().tm_year
    data['Edad'] = current_year - data['Año']
    data['Km promedio por año'] = data['Kilómetros'] // data['Edad'].replace(0, 1)
    return data

# --------- Clean year column ----------

def clean_year(data, filename):
    outliers = data.loc[data['Año'] > time.localtime().tm_year].index
    outliers = outliers.sort_values(ascending=False)
    delete_rows(data, outliers, filename)

# --------- Clean door column ----------

def clean_doors(data, filename):
    outliers = data.loc[data['Puertas'] > 5 or data['Puertas'] < 4].index
    outliers = outliers.sort_values(ascending=False)
    delete_rows(data, outliers, filename)


# --------- Generic functions ----------
    
def find_outliers(data, feature, threshold):
    num_outliers = []
    for i in range(len(threshold)):
           outliers = data.loc[data[feature] >= threshold[i]] 
           num_outliers.append(outliers.shape[0])
    return num_outliers, threshold