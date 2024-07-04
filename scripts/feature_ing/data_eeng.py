import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

def create_new_features(data):
    # create_new_features
    # Función que crea nuevas features a partir en base a transformaciones lineales entre otras features existentes.
    # Parámetros:
    # - data: DataFrame con los datos limpiados.
    # Retorna:
    # - df: DataFrame con las nuevas features añadidas.
    
    # Copiar el dataframe original para no modificar el original
    df = data.copy()
    
    # Características binarias
    df['High_Mileage'] = (df['Kilómetros'] > 100000).astype(int)
    df['Is_Old'] = (df['Edad'] > 10).astype(int)

    # Ratios
    df['Engine_Size_per_Year'] = df['Motor'] / df['Edad'].replace(0, np.nan)
    df['Km_per_Cylinder'] = df['Kilómetros'] / df['Cilindros'].replace(0, np.nan)

    # Interacciones entre características
    df['Motor_Age_Interaction'] = df['Motor'] * df['Edad']
    df['Mileage_Turbo_Interaction'] = df['Kilómetros'] * df['Turbo']
    df['Motor_Cylinder_Interaction'] = df['Motor'] * df['Cilindros']

    # Reemplazar valores infinitos y NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    # Transformaciones polinomiales
    numeric_features = df[['Año', 'Kilómetros', 'Edad', 'Km promedio por año', 'Cilindros', 'Motor']]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(numeric_features)
    poly_feature_names = poly.get_feature_names_out(numeric_features.columns)

    df_poly = pd.DataFrame(poly_features, columns=poly_feature_names)
    
    df_poly = df_poly.drop(columns=['Año', 'Kilómetros', 'Edad', 'Km promedio por año', 'Cilindros', 'Motor'])
    # Concatenar las características polinomiales al dataframe original
    df = pd.concat([df, df_poly], axis=1) 
    
    return df