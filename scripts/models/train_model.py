import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Añadir el directorio raíz del proyecto al path para importar módulos
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

log_dir = os.path.join(project_root, 'logs')
# Configuración de logging
logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


from src.data.load_data import load_data

def get_model(model_name):
    if model_name == 'linear_regression':
        from src.models.linear_regression import LinearRegressionModel
        return LinearRegressionModel()
    elif model_name == 'random_forest':
        from src.models.random_forest import RandomForestModel
        return RandomForestModel()
    elif model_name == 'xgboost':
        from src.models.xgboost import XGBoostModel
        return XGBoostModel()
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

def main(model_name):
    filepath = os.path.join(project_root, 'data/data.csv')  # Ruta al archivo CSV
    logging.info(f"Cargando datos desde {filepath}")
    data = load_data(filepath)
    
    X = data.drop(columns=['Precio'])
    y = data['Precio']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logging.info(f"Entrenando modelo {model_name}")
    model = get_model(model_name)
    model.train(X_train, y_train)
    
    mae, mse, rmse, r2 = model.evaluate(X_test, y_test)
    logging.info(f"Resultados del modelo {model_name} - MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R^2: {r2}")
    
    print(f"Modelo: {model_name}")
    print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R^2: {r2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar un modelo de ML especificado.")
    parser.add_argument('model', type=str, help="El nombre del modelo a usar (e.g., 'linear_regression', 'random_forest', 'decision_tree', 'xgboost')")
    args = parser.parse_args()
    main(args.model)
