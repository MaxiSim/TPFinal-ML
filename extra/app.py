import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Ruta del modelo
model_path = '/home/agustin/Documentos/3ro/ML&DL/TrabajoFinal/TPFinal-ML/src/models/saved/xg.joblib'
model = joblib.load(model_path)


# Título de la aplicación
st.title('Predicción de Precios de Vehículos')

# Formulario de entrada
st.header('Ingresa las características del vehículo')
modelo = st.selectbox('Modelo', ['Modelo1', 'Modelo2', 'Modelo3'])
año = st.number_input('Año', min_value=2000, max_value=2024, value=2015)
color = st.selectbox('Color', ['Rojo', 'Azul', 'Negro', 'Blanco'])
motor = st.selectbox('Motor', ['1.0L', '1.5L', '2.0L'])
kilometros = st.number_input('Kilómetros', min_value=0, value=50000)
edad = st.number_input('Edad', min_value=0, value=5)
km_promedio_por_año = st.number_input('Km promedio por año', min_value=0, value=10000)
cilindros = st.number_input('Cilindros', min_value=2, max_value=12, value=4)
turbo = st.selectbox('Turbo', ['No', 'Sí'])

# Preprocesar entrada
input_data = pd.DataFrame({
    'Modelo': [modelo],
    'Año': [año],
    'Color': [color],
    'Motor': [motor],
    'Kilómetros': [kilometros],
    'Edad': [edad],
    'Km promedio por año': [km_promedio_por_año],
    'Cilindros': [cilindros],
    'Turbo': [1 if turbo == 'Sí' else 0]
})

def preprocess_input(data):
    # Aquí debes incluir tu lógica de preprocesamiento
    # Por ejemplo, transformar variables categóricas a numéricas, etc.
    data = pd.get_dummies(data)
    return data

input_data_processed = preprocess_input(input_data)

# Ajustar columnas faltantes
expected_columns = ['Modelo_Modelo1', 'Modelo_Modelo2', 'Modelo_Modelo3', 'Año', 'Color_Rojo', 'Color_Azul', 'Color_Negro', 'Color_Blanco', 'Motor_1.0L', 'Motor_1.5L', 'Motor_2.0L', 'Kilómetros', 'Edad', 'Km promedio por año', 'Cilindros', 'Turbo']
input_data_processed = input_data_processed.reindex(columns=expected_columns, fill_value=0)

# Realizar la predicción
if st.button('Predecir Precio'):
    prediction = model.predict(input_data_processed)
    st.write(f'El precio estimado del vehículo es: ${prediction[0]:,.2f}')
