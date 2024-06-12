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

def delete_rows(filename, index):
    data = read_data(filename)
    data = data.drop(index)
    rewrite_data(data, filename)
    
def find_outliers(data, feature):
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data[feature] < (Q1 - 1.5 * IQR)) | (data[feature] > (Q3 + 1.5 * IQR))]
    return outliers

def process_engine(filename):
    data = read_data(filename)

    max_engine = float(data['Motor'].str.extract(r'(\d+\.\d+)').dropna().max())
    min_engine = float(data['Motor'].str.extract(r'(\d+\.\d+)').dropna().min())

    print(f'Min engine: {min_engine}')
    print(f'Max engine: {max_engine}')
    
    kernels = [] 
    linespace = np.arange(min_engine, max_engine, 0.1)
    for i in linespace:
        kernels.append(round(i, 1))

    for i in range(len(kernels)):
        kernels[i] = str(kernels[i])

    print(f'Kernels: {kernels}')

    muestras_sin_motor = []

    for j in range(len(data)):
        motor = str(data.loc[j, 'Motor'])
        version = str(data.loc[j, 'Versión'])

        found = False
        for kernel in kernels:
            if convolucion_motor(motor, kernel) or convolucion_motor(version, kernel):
                data.loc[j, 'Motor'] = kernel
                found = True
                break

        if not found:
            print(f'No se encontró motor para la muestra {j} con motor {motor} y versión {version}')
            muestras_sin_motor.append(j)        
    print(f'Muestras sin motor: {muestras_sin_motor}')
    print(f'Cantidad de muestras sin motor: {len(muestras_sin_motor)}')

        
def convolucion_motor(version, kernel):
    if kernel in version:
        return True
    return False
   
filename="data/pf_suvs_i302_1s2024.csv"

process_engine(filename=filename)

