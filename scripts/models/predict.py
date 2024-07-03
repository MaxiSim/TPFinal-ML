import os
import sys
import pandas as pd

# Agregar la ruta al directorio raíz del proyecto para importar los módulos
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from src.data.load_data import load_data