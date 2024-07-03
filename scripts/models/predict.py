import os
import sys
import argparse
import logging

# Añadir el directorio raíz del proyecto al path para importar módulos
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

# Configuración de logging
log_dir = os.path.join(project_root, 'logs')
logging.basicConfig(filename=os.path.join(log_dir, 'evaluate_test.log'), level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

from src.data.load_data import load_data
from src.utils.utils import *
import csv

def predict(model_type, model_name):
    # main
    # Función principal que evalúa un modelo de ML especificado.
    # El modelo a evaluar debe ser un archivo .joblib guardado en la carpeta src/models/saved.
    # Se deben especificar el tipo de modelo y el nombre del archivo .joblib.
    # Parámetros:
    # - args: Argumentos de línea de comandos.


    filepath = os.path.join(project_root, 'data/BOOST_TEST_DATASET.csv')  # Ruta al archivo CSV
    logging.info(f"Cargando datos desde {filepath}")
    data = load_data(filepath)
    
    logging.info(f"Cargando y evaluando un {model_type} desde models/saved/{model_name}.joblib")
    model = get_model(model_type) # Seleccionar el modelo a usar 

    try:
        model.load_model(os.path.join(project_root, 'src/models/saved', f'{model_name}.joblib'))
    except FileNotFoundError:
        logging.error(f"El archivo {model_name}.joblib no existe en {os.path.join(project_root, 'src/models/saved')}")
        sys.exit(1)

    y_pred = model.predict(data)
    output_file = os.path.join(project_root, 'predictions_Simian_Manzano.csv')
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'Predicted_Price_USD'])  # Write header
        for i, pred in enumerate(y_pred, start=1):
            writer.writerow([i, pred])
    logging.info(f"Predictions saved in {output_file}")



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Evaluar un modelo de ML especificado.")
#     parser.add_argument('--type', type=str, help="El tipo de modelo a usar (e.g., 'linear_regression', 'random_forest', 'xgboost')")
#     parser.add_argument('saved_model_name', type=str, help="El nombre del modelo guardado (sin la extensión .joblib)")
#     args = parser.parse_args()
#     main(args)
