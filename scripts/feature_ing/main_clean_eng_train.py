import pandas as pd
import os
import sys
# Añadir el directorio raíz del proyecto al path para importar módulos
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from data_cleaner import DataCleaner
from src.data.load_data import load_data
from data_eeng import create_new_features
from oversampling import oversampling


def main():
    # Funcion que llama al DataCleaner para limpiar los datos y luego crea nuevas fratures.
    # Finalmente, se realiza oversampling para balancear las clases.
    # Genera un nuevo dataset procesado y con las nuevas características.

    # Añadir el directorio raíz del proyecto al path para importar módulos
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    sys.path.append(project_root)
    
    data = load_data(os.path.join(project_root, 'data/pf_suvs_i302_1s2024.csv'))
    cleaner = DataCleaner(data, project_root)

    data = pd.read_csv(os.path.join(project_root, 'data/CLEAN_APP_DATASET.csv'))
    # data = create_new_features(data)
    # data = oversampling(data)

    # # # Guardar el nuevo dataset con las características generadas
    # data.to_csv(os.path.join(project_root, 'data/BOOST_TRAIN_DATASET.csv'), index=False)
    

if __name__ == '__main__':
    main()
    