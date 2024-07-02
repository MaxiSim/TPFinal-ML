import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Agregar la ruta al directorio raíz del proyecto para importar los módulos
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from src.models.linear_regression import LinearRegressionModel
from src.data.load_data import load_data

# Cargar los datos
data = load_data(os.path.join(project_root, 'data', 'data.csv'))

# Separar características y objetivo
X = data.drop(columns=['Precio'])  # Asume que la columna 'price' es el objetivo
y = data['Precio']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo y cargar el modelo guardado
lr_model = LinearRegressionModel()
lr_model.load_model(os.path.join(project_root, 'outputs', 'models', 'linear_regression_model.pkl'))

# Evaluar el modelo
mae, mse, rmse, r2 = lr_model.evaluate(X_test, y_test)
print(f'MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R^2: {r2}')
