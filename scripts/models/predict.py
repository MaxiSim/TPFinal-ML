import os
import sys
import pandas as pd

# Agregar la ruta al directorio raíz del proyecto para importar los módulos
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from src.models.linear_regression import LinearRegressionModel
from src.data.load_data import load_data

# Función para cargar nuevos datos para predicción
def load_new_data(filepath):
    # Aquí puedes definir cómo cargar y preprocesar los nuevos datos para predicción
    # Por ejemplo, podrías cargar un CSV similar a cómo se hace en load_data
    data = pd.read_csv(filepath)
    # Realiza el preprocesamiento necesario aquí
    return data

# Cargar nuevos datos para predicción
new_data = load_new_data(os.path.join(project_root, 'data', 'new_data.csv'))

# Separar características
X_new = new_data.drop(columns=['Precio'])  # Si tienes el precio en los nuevos datos, exclúyelo

# Crear el modelo y cargar el modelo guardado
lr_model = LinearRegressionModel()
lr_model.load_model(os.path.join(project_root, 'outputs', 'models', 'linear_regression_model.pkl'))

# Realizar predicciones
predictions = lr_model.predict(X_new)
new_data['predicted_price'] = predictions

# Guardar los resultados
new_data.to_csv(os.path.join(project_root, 'outputs', 'predictions', 'predicted_prices.csv'), index=False)
print('Predicciones guardadas en ../../outputs/predictions/predicted_prices.csv')
