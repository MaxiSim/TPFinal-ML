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

    # print(f'Min engine: {min_engine}')
    # print(f'Max engine: {max_engine}')
    
    kernels = [] 
    linespace = np.arange(min_engine, max_engine+0.1, 0.1)
    for i in linespace:
        kernels.append(round(i, 1))

    for i in range(len(kernels)):
        kernels[i] = str(kernels[i])

    # print(f'Kernels: {kernels}')

    muestras_sin_motor = []

    for j in range(len(data)):
        motor = str(data.loc[j, 'Motor'])
        version = str(data.loc[j, 'Versión'])

        found = False
        for kernel in kernels:
            if convolucion_1d(motor, kernel) or convolucion_1d(version, kernel):
                data.loc[j, 'Motor'] = kernel
                found = True
                break

        if not found:
            if str(data.loc[j, 'Tipo de combustible']) not in ['Híbrido', 'Eléctrico']:
                # print(f'No se encontró motor para la muestra {j} con motor {motor} y versión {version}')
                muestras_sin_motor.append(j)        
    # print(f'Muestras sin motor: {muestras_sin_motor}')
    # print(f'Cantidad de muestras sin motor: {len(muestras_sin_motor)}')
    return muestras_sin_motor

        
def convolucion_1d(feature, kernel):
    if kernel.lower() in feature.lower():
        return True
    return False


def read_data(filename):
    # Supuesto método para leer datos
    return pd.read_csv(filename)



def process_color(filename):
    data = read_data(filename)
    muestras_sin_color = []
    
    color_mappings = {
        'blanc': 'blanco',
        'negr': 'negro',
        'plat': 'plata', 'silver': 'plata',
        'gr': 'gris',
        'rojo': 'rojo',
        'bordó': 'bordó',
        'verde': 'verde',
        'azul': 'azul', 'blue': 'azul',
        'a ele': 'elección', 'todos': 'elección',
        'rosa': 'rosa',
        'violeta': 'violeta',
        'celeste': 'celeste',
        'dorado': 'dorado',
        'amarillo': 'amarillo',
        'naranja': 'naranja', 'orange': 'naranja',
        'marrón': 'marrón', 'ocre': 'marrón', 'beige': 'marrón'
    }
    
    for i in range(len(data)):
        color = str(data.loc[i, 'Color']).lower()
        found_color = False
        
        for pattern, new_color in color_mappings.items():
            if convolucion_1d(color, pattern):
                data.loc[i, 'Color'] = new_color
                found_color = True
                break
        
        if not found_color:
            muestras_sin_color.append(i)
    
    print(f'Despues del primer filtro, las muestras con color quedaron así:')
    print(data['Color'].value_counts())
    print(f'Muestras sin color: {data.loc[muestras_sin_color, "Color"].value_counts()}')


    # Distribución de colores clasificados
    color_distribution = data.loc[~data.index.isin(muestras_sin_color), 'Color'].value_counts(normalize=True)
    
    # Rellenar los casos sin color basado en la distribución
    colors = color_distribution.index.tolist()
    probabilities = color_distribution.values
    
    # Asignar colores aleatoriamente a las muestras sin color
    data.loc[muestras_sin_color, 'Color'] = np.random.choice(colors, size=len(muestras_sin_color), p=probabilities)
    
    print(f'Muestras sin color después de asignación: {data.loc[muestras_sin_color, "Color"].value_counts()}')
    print(f'Distribución final de colores: {data["Color"].value_counts()}')
    print(f'Finalmente, se asignaron colores a {len(muestras_sin_color)} muestras sin color')
    
    #elimino las muestras que no tienen color
    # data = data.drop(muestras_sin_color)              ######### PREGUNTAR A MAXI ##########
    # print(f'Cantidad de muestras sin color: {data["Color"].isna().sum()}')


    return data

def process_color_indef(filename):
    data = read_data(filename)
    
    color_mappings = {
        'blanc': 'blanco',
        'negr': 'negro',
        'plat': 'plata', 'silver': 'plata',
        'gr': 'gris',
        'rojo': 'rojo',
        'bordó': 'bordó',
        'verde': 'verde',
        'azul': 'azul', 'blue': 'azul',
        'a ele': 'elección', 'todos': 'elección',
        'rosa': 'rosa',
        'violeta': 'violeta',
        'celeste': 'celeste',
        'dorado': 'dorado',
        'amarillo': 'amarillo',
        'naranja': 'naranja', 'orange': 'naranja',
        'marrón': 'marrón', 'ocre': 'marrón', 'beige': 'marrón'
    }
    
    muestras_sin_color = []
    for i in range(len(data)):
        color = str(data.loc[i, 'Color']).lower()
        found_color = False
        
        for pattern, new_color in color_mappings.items():
            if convolucion_1d(color, pattern):
                data.loc[i, 'Color'] = new_color
                found_color = True
                break
        
        if not found_color:
            muestras_sin_color.append(i)
    
    print(f'Despues del primer filtro, las muestras con color quedaron así:')
    print(data['Color'].value_counts())
    print(f'Muestras sin color: {data.loc[muestras_sin_color, "Color"].value_counts()}')
    print(f'Cantidad de muestras nan: {data["Color"].isna().sum()}')
    #a las muestras que son nan les asigno un color 'indefinido'
    data.loc[data['Color'].isna(), 'Color'] = 'indefinido'
    print(f'Finalmente, los colores quedaron así:')
    print(data['Color'].value_counts())

    return data


def process_cylinders(filename):
    data = read_data(filename)
    
    # Buscar en la columna 'Versión' si contiene 'v6' o 'v8'
    data['cilindros'] = data['Versión'].apply(lambda x: 2 if convolucion_1d(str(x), 'v6') else (3 if convolucion_1d(str(x), 'v8') else 1))

    print(f'Cantidad de cilindros: {data["cilindros"].value_counts()}')

    return data


def process_turbo(filename):
    
    data = read_data(filename)

    data['turbo'] = data['Versión'].apply(lambda x: 1 if convolucion_1d(str(x), 'turbo') else 0)

    print(f'Cantidad de autos con turbo: {data["turbo"].value_counts()}')

    return data

# def process_version(filename):
    
#     data = read_data(filename)

#     marca = random.choice(data['Marca'].unique())
#     modelo = random.choice(data.loc[data['Marca'] == marca]['Modelo'].unique())
#     # anio = random.choice(data.loc[(data['Marca'] == marca) & (data['Modelo'] == modelo)]['Año'].unique())
#     versiones = data.loc[(data['Marca'] == marca) & (data['Modelo'] == modelo)]['Versión']
#     print(f'Versiones del modelo {modelo}: {versiones.unique()}')    
#     precios_por_cada_version = []

#     for version in versiones:
#         precios = data.loc[(data['Marca'] == marca) & (data['Modelo'] == modelo) & (data['Versión'] == version)]['Precio'].values
#         precios_por_cada_version.append(precios)
#         print(f'Precios de la versión {version}: {precios}')
        
#     fig = go.Figure()
#     for i in range(len(versiones)):
#         fig.add_trace(go.Scatter(x=data.loc[(data['Marca'] == marca) & (data['Modelo'] == modelo) & (data['Versión'] == versiones.iloc[i])]['Año'], y=precios_por_cada_version[i], mode='markers', name=versiones.iloc[i]))
#     fig.update_layout(title_text=f'Prices of versions of model {modelo} ', xaxis_title_text='Year', yaxis_title_text='Price')
#     fig.show()    



filename = 'data/pf_suvs_i302_1s2024.csv'

process_color(filename)
