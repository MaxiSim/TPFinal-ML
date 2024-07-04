import streamlit as st
import pandas as pd
import os
import sys
from joblib import load
from utils import input_to_features, get_model_value

# Ruta del modelo
# Añadir el directorio raíz del proyecto al path para importar módulos
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from os.path import dirname, abspath
d = dirname(dirname(abspath(__file__)))
sys.path.append(d)


# Cargar modelo
model_path = os.path.join(project_root, 'TPFinal-ML/src/models/saved/xg.joblib')
model = load(model_path)

# Diccionario de marcas y modelos
# primero, cargo el dataset
dataset_path = os.path.join(project_root, 'TPFinal-ML/data/pf_suvs_i302_1s2024.csv')
df = pd.read_csv(dataset_path)

# Crear un diccionario con las marcas y modelos
car_data = {}
for marca in df['Marca'].unique():
    car_data[marca] = df[df['Marca'] == marca]['Modelo'].unique()

# Crear un diccionario con los modelos y el tipo de combustible
model_combustible_data = {}
for modelo in df['Modelo'].unique():
    model_combustible_data[modelo] = df[df['Modelo'] == modelo]['Tipo de combustible'].unique()

# Crear un diccionario con los modelos y el tipo de motor
model_motor_data = {}
for modelo in df['Modelo'].unique():
    model_motor_data[modelo] = df[df['Modelo'] == modelo]['Motor'].unique()

# Crear un diccionario con los modelos y el color
model_color_data = {}
for modelo in df['Modelo'].unique():
    model_color_data[modelo] = df[df['Modelo'] == modelo]['Color'].unique()

# Crear la interfaz de Streamlit
st.title('Predicción de Precios de SUVs')

# Selección de las características del coche
marca = st.selectbox('Marca', options=list(car_data.keys()))
modelo = st.selectbox('Modelo', options=car_data[marca])
motor = st.selectbox('Motor', options=model_motor_data[modelo])
kilometraje = st.number_input('Kilometraje', min_value=0, step=1)
año = st.number_input('Año', min_value=1990, max_value=2024, step=1)
color = st.selectbox('Color', options=model_color_data[modelo])
combustible = st.selectbox('Combustible', options=model_combustible_data[modelo])
transmision = st.selectbox('Transmisión', options=['Manual', 'Automática'])

# Crear un diccionario con las características de entrada
input_features = {
    'Marca': marca,
    'Modelo': modelo,
    'Motor': motor,
    'Kilómetros': kilometraje,
    'Año': año,
    'Color': color,
    'Tipo de combustible': combustible,
    'Transmisión': transmision
}

# Convertir las características de entrada a un DataFrame
input_data = pd.DataFrame([input_features])

# Mostrar las características de entrada
st.write('Características de Entrada:')
st.write(input_data)


# Botón para realizar la predicción
if st.button('Realizar Predicción'):
    # Transformar los datos de entrada
    input_data = input_to_features(input_data)

    # Realizar la predicción
    prediction = model.predict(input_data)
    
    # Mostrar el resultado de la predicción
    st.write('Predicción del Precio del SUV:')
    st.write(prediction[0])

