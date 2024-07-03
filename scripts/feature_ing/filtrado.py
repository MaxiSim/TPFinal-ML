import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random
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


def convolucion_1d(feature, kernel):
    if kernel.lower() in feature.lower():
        return True
    return False
        

## --------- Clean color ----------


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
    
    # print(f'Despues del primer filtro, las muestras con color quedaron así:')
    # print(data['Color'].value_counts())
    # print(f'Muestras sin color: {data.loc[muestras_sin_color, "Color"].value_counts()}')


    # Distribución de colores clasificados
    color_distribution = data.loc[~data.index.isin(muestras_sin_color), 'Color'].value_counts(normalize=True)
    
    # Rellenar los casos sin color basado en la distribución
    colors = color_distribution.index.tolist()
    probabilities = color_distribution.values
    
    # Asignar colores aleatoriamente a las muestras sin color
    data.loc[muestras_sin_color, 'Color'] = np.random.choice(colors, size=len(muestras_sin_color), p=probabilities)
    
    # print(f'Muestras sin color después de asignación: {data.loc[muestras_sin_color, "Color"].value_counts()}')
    # print(f'Distribución final de colores: {data["Color"].value_counts()}')
    # print(f'Finalmente, se asignaron colores a {len(muestras_sin_color)} muestras sin color')
    
    #elimino las muestras que no tienen color
    data = data.drop(muestras_sin_color)              ######### PREGUNTAR A MAXI ##########
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


## --------- Process engine spects ----------

def process_cylinders(filename):
    data = read_data(filename)
    
    # Buscar en la columna 'Versión' si contiene 'v6' o 'v8'
    data['Cilindros'] = data['Versión'].apply(lambda x: 2 if convolucion_1d(str(x), 'v6') else (3 if convolucion_1d(str(x), 'v8') else 1))

    # print(f'Cantidad de cilindros: {data["Cilindros"].value_counts()}')

    return data


def process_turbo(filename):
    
    data = read_data(filename)

    data['Turbo'] = data['Versión'].apply(lambda x: 1 if convolucion_1d(str(x), 'turbo') else 0)

    # print(f'Cantidad de autos con turbo: {data["Turbo"].value_counts()}')

    return data


def process_engine(filename):
    data = read_data(filename)

    max_engine = float(data['Motor'].str.extract(r'(\d+\.\d+)').dropna().max())
    min_engine = float(data['Motor'].str.extract(r'(\d+\.\d+)').dropna().min())

    kernels = [] 
    linespace = np.arange(min_engine, max_engine+0.1, 0.1)
    for i in linespace:
        kernels.append(str(round(i, 1)))


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
                data.loc[j, 'Motor'] = 'No especificado'
            else:
                data.loc[j, 'Motor'] = 'No aplica'

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



# --------- Encoding categorical features ----------

def encode_categorical(data, feature):
    data[feature] = data[feature].astype('category')
    data[feature] = data[feature].cat.codes

    return data


def one_hot_encoding(data, feature):
    data = pd.get_dummies(data, columns=[feature])
    return data

def target_encoding(data, feature, target):
    data[feature] = data.groupby(feature)[target].transform('mean').astype(int)
    return data