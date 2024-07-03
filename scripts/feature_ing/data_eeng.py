import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import os
import sys
import numpy as np

def create_new_features(data):
    # Copiar el dataframe original para no modificar el original
    df = data.copy()
    
    # Características binarias
    df['High_Mileage'] = (df['Kilómetros'] > 100000).astype(int)
    df['Is_Old'] = (df['Edad'] > 10).astype(int)

    # Ratios
    df['Engine_Size_per_Year'] = df['Motor'] / df['Edad'].replace(0, np.nan)
    df['Km_per_Cylinder'] = df['Kilómetros'] / df['Cilindros'].replace(0, np.nan)
    df['Price_per_Km'] = df['Precio'] / df['Kilómetros'].replace(0, np.nan)

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
    
    # Concatenar las características polinomiales al dataframe original
    df = pd.concat([df, df_poly], axis=1)
    
    return df


def oversampling(data):
    model_counts = data['Modelo'].value_counts()

    for model, count in model_counts.items():
        if count < 1200:
            duplicates_needed = 1500 - count
            model_samples = data[data['Modelo'] == model]
            duplicated_samples = model_samples.sample(n=duplicates_needed, replace=True)
            data = pd.concat([data, duplicated_samples])
    
    return data


# Añadir el directorio raíz del proyecto al path para importar módulos
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

data = pd.read_csv(os.path.join(project_root, 'data/CLEAN_TRAIN_DATASET.csv'))
data = create_new_features(data)
data = oversampling(data)

# Guardar el nuevo dataset con las características generadas
data.to_csv(os.path.join(project_root, 'data/BOOST_DATASET.csv'), index=False)
