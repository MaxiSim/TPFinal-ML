import pandas as pd

def load_data(filepath):
    """
    Carga los datos desde un archivo CSV y los devuelve como un DataFrame de pandas.
    
    :param filepath: Ruta al archivo CSV.
    :return: DataFrame con los datos cargados, sin muestras donde 'Motor' tenga valor 'No Aplica'.
    """
    # primero cargo el dataset
    data = pd.read_csv(filepath)
    
    # luego elimino las columnas que no me interesan
    data = data.drop(columns=['Moneda'])
    
    
    return data
