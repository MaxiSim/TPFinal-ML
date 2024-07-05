import os
import sys
import argparse
import logging
from sklearn.model_selection import train_test_split

# Añadir el directorio raíz del proyecto al path para importar módulos
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

# Configuración de logging
log_dir = os.path.join(project_root, 'logs')
logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

from src.data.load_data import load_data
from src.utils.utils import get_model, plot_feature_importances

def main(args):
    # main
    # Función principal que entrena un modelo de ML especificado y evalúa su rendimiento.
    # El modelo entrenado puede ser de tipo 'linear_regression', 'random_forest' o 'xgboost'.
    # Se pueden optimizar los hiperparámetros del modelo con la opción --optimize.
    # Se puede guardar el modelo entrenado con la opción --save_model, especificando un nombre con --name.
    # Parámetros:
    # - args: Argumentos de línea de comandos.
    
    model_type = args.model
    
    filepath = os.path.join(project_root, 'data/BOOST_TRAIN_DATASET.csv')  # Ruta al archivo CSV
    logging.info(f"Cargando datos desde {filepath}")
    data = load_data(filepath)
    
    X = data.drop(columns=['Precio'])
    y = data['Precio']

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, y_train = X, y
    
    logging.info(f"Entrenando modelo {model_type}")
 
    if model_type == 'nn':
        if args.optimize:
            model, params_opt = get_model(model_type, optimize=True, X_train=X_train, y_train=y_train)
            logging.info(f"Mejores hiperparámetros encontrados: {params_opt}")
        else:
            model = get_model(model_type, input_dim=X_train.shape[1])
    else:
        if args.optimize:
            model, params_opt = get_model(model_type, optimize=True, X_train=X_train, y_train=y_train)
            logging.info(f"Mejores hiperparámetros encontrados: {params_opt}")
        else:
            model = get_model(model_type)
    
    model.train(X_train, y_train)
    
    mae, mse, rmse, r2 = model.evaluate(X_train, y_train)
    logging.info(f"Resultados del modelo {model_type} - MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R^2: {r2}")
    
    print(f"Modelo: {model_type}")
    print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R^2: {r2}")

    if args.save_model:
        model.save_model(os.path.join(project_root, 'src/models/saved', f'{args.name}.joblib'))
        logging.info(f"Modelo guardado en {os.path.join(project_root, 'src/models/saved', f'{model_type}.joblib')}")

    importances = model.feature_importances_

    plot_feature_importances(importances, X.columns)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar un modelo de ML especificado.")
    parser.add_argument('model', type=str, help="El nombre del modelo a usar (e.g., 'linear_regression', 'random_forest', 'xgboost', 'nn')")
    parser.add_argument('--optimize', action='store_true', help="Optimizar los hiperparámetros del modelo")
    parser.add_argument('--save_model', action='store_true', help="Guardar el modelo entrenado")
    parser.add_argument('--name', type=str, help="El nombre del archivo del modelo guardado")
    args = parser.parse_args()

    if args.optimize and args.model == 'linear_regression':
        parser.error("No se pueden optimizar los hiperparámetros de un modelo de regresión lineal")
    if args.optimize and args.model == 'nn':
        parser.error("No esta soportada la optimización de hiperparámetros para el modelo de red neuronal")
    if args.save_model and not args.name:
        parser.error("--save_model requiere --name para especificar el nombre del archivo del modelo guardado")

    main(args)
