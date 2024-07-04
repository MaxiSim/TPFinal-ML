import pandas as pd
import os
import sys
# Añadir el directorio raíz del proyecto al path para importar módulos
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from scripts.feature_ing.data_cleaner import DataCleaner
from src.data.load_data import load_data
from scripts.feature_ing.data_eeng import create_new_features


def process_test_data(filename):
    # Funcion que llama al DataCleaner para limpiar los datos y luego crea nuevas fratures.
    # Genera un nuevo dataset procesado y con las nuevas características.

    # Añadir el directorio raíz del proyecto al path para importar módulos
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    sys.path.append(project_root)
    
    data = load_data(os.path.join(project_root, filename))
    cleaner = DataCleaner(data, project_root, eval_data=True) 

    data = pd.read_csv(os.path.join(project_root, 'data/CLEAN_TEST_DATASET.csv'))
    data = create_new_features(data)

    data = data[['Modelo', 'Año', 'Color', 'Motor', 'Kilómetros', 'Edad', 'Km promedio por año', 'Cilindros', 'Turbo', 'Transmisión_Automática', 'Transmisión_Automática secuencial', 'Transmisión_Manual', 'Transmisión_Semiautomática', 'Marca_Abarth', 'Marca_Alfa Romeo', 'Marca_Audi', 'Marca_BAIC', 'Marca_BMW', 'Marca_Chery', 'Marca_Chevrolet', 'Marca_Citroën', 'Marca_DS', 'Marca_Daihatsu', 'Marca_Dodge', 'Marca_Fiat', 'Marca_Ford', 'Marca_Geely', 'Marca_Haval', 'Marca_Honda', 'Marca_Hyundai', 'Marca_Isuzu', 'Marca_JAC', 'Marca_Jaguar', 'Marca_Jeep', 'Marca_Jetour', 'Marca_Kia', 'Marca_Land Rover', 'Marca_Lexus', 'Marca_Lifan', 'Marca_MINI', 'Marca_Mercedes-Benz', 'Marca_Mitsubishi', 'Marca_Nissan', 'Marca_Peugeot', 'Marca_Porsche', 'Marca_Renault', 'Marca_Sandero', 'Marca_Ssangyong', 'Marca_Subaru', 'Marca_Suzuki', 'Marca_Toyota', 'Marca_Volkswagen', 'Marca_Volvo', 'Tipo de combustible_Diésel', 'Tipo de combustible_Eléctrico', 'Tipo de combustible_GNC', 'Tipo de combustible_Híbrido', 'Tipo de combustible_Híbrido/Diesel', 'Tipo de combustible_Híbrido/Nafta', 'Tipo de combustible_Nafta', 'Tipo de combustible_Nafta/GNC', 'Tipo de vendedor_concesionaria', 'Tipo de vendedor_particular', 'Tipo de vendedor_tienda', 'High_Mileage', 'Is_Old', 'Engine_Size_per_Year', 'Km_per_Cylinder', 'Motor_Age_Interaction', 'Mileage_Turbo_Interaction', 'Motor_Cylinder_Interaction', 'Año^2', 'Año Kilómetros', 'Año Edad', 'Año Km promedio por año', 'Año Cilindros', 'Año Motor', 'Kilómetros^2', 'Kilómetros Edad', 'Kilómetros Km promedio por año', 'Kilómetros Cilindros', 'Kilómetros Motor', 'Edad^2', 'Edad Km promedio por año', 'Edad Cilindros', 'Edad Motor', 'Km promedio por año^2', 'Km promedio por año Cilindros', 'Km promedio por año Motor', 'Cilindros^2', 'Cilindros Motor', 'Motor^2']]
    
    data.to_csv(os.path.join(project_root, 'data/BOOST_TEST_DATASET.csv'), index=False)
    
    
    
# 'data/pf_suvs_test_ids_i302_1s2024.csv'