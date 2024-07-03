import pandas as pd

def oversampling(data):
    # oversampling
    # Función que realiza oversampling en el dataset para balancear las clases.
    # Duplica muestras al azar de las clases minoritarias hasta que todas tengan 1500 muestras.
    # Parámetros:
    # - data: DataFrame con los datos limpiados.
    # Retorna:
    # - data: DataFrame con las muestras duplicadas.
    
    model_counts = data['Modelo'].value_counts()

    for model, count in model_counts.items():
        if count < 1200:
            duplicates_needed = 1500 - count
            model_samples = data[data['Modelo'] == model]
            duplicated_samples = model_samples.sample(n=duplicates_needed, replace=True)
            data = pd.concat([data, duplicated_samples])
    
    return data