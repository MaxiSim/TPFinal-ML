#!/bin/bash

# Activar el entorno virtual si es necesario
# source /ruta/a/tu/entorno_virtual/bin/activate

# Definir la ruta al directorio donde se encuentra main.py
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Cambiar al directorio donde se encuentra main.py
cd "$DIR"

# Ejecutar el script Python
python main.py

# Desactivar el entorno virtual si es necesario
# deactivate
