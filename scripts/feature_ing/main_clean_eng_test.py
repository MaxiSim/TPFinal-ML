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


def process_test_data():
    # Funcion que llama al DataCleaner para limpiar los datos y luego crea nuevas fratures.
    # Genera un nuevo dataset procesado y con las nuevas características.

    # Añadir el directorio raíz del proyecto al path para importar módulos
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    sys.path.append(project_root)
    
    data = load_data(os.path.join(project_root, 'data/pf_suvs_test_ids_i302_1s2024.csv'))
    cleaner = DataCleaner(data, project_root, eval_data=True) 

    data = pd.read_csv(os.path.join(project_root, 'data/CLEAN_TRAIN_DATASET.csv'))
    data = create_new_features(data)

    # Guardar el nuevo dataset con las características generadas
    data.to_csv(os.path.join(project_root, 'data/BOOST_TEST_DATASET.csv'), index=False)
    


process_test_data()