import streamlit as st
import pandas as pd
import os
import sys
from joblib import load
import plotly.express as px
from utils import input_to_features, get_model_value

# Añadir el directorio raíz del proyecto al path para importar módulos
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

# Cargar modelo
model_path = os.path.join(project_root, 'TPFinal-ML/src/models/saved/xg_opt_over.joblib')
model = load(model_path)

# Diccionario de marcas y modelos
dataset_path = os.path.join(project_root, 'TPFinal-ML/data/CLEAN_APP_DATASET.csv')
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

# Crear un diccionario con los modelos y los años
model_year_data = {}
for modelo in df['Modelo'].unique():
    model_year_data[modelo] = df[df['Modelo'] == modelo]['Año'].unique().round().astype(int)

# Crear un diccionario con los modelos y las transmisiones
model_transmision_data = {}
for modelo in df['Modelo'].unique():
    model_transmision_data[modelo] = df[df['Modelo'] == modelo]['Transmisión'].unique()

# Crear la interfaz de Streamlit
st.set_page_config(page_title='Predicción de Precios de SUVs', page_icon='🚙')
st.title('Predicción de Precios de SUVs')

# Sidebar para opciones adicionales
st.sidebar.title('Opciones')
show_data_info = st.sidebar.checkbox('Mostrar Información de Datos')
show_explanation = st.sidebar.checkbox('Mostrar Explicación')

# Sección principal de la aplicación
with st.expander('Configuración de SUV'):
    # Selección de las características del coche
    marca = st.selectbox('Marca', options=list(car_data.keys()))
    modelo = st.selectbox('Modelo', options=car_data[marca])
    motor = st.selectbox('Motor', options=model_motor_data[modelo])
    kilometraje = st.number_input('Kilometraje', min_value=0, step=1)
    año = st.selectbox('Año', options=model_year_data[modelo])
    color = st.selectbox('Color', options=model_color_data[modelo])
    combustible = st.selectbox('Combustible', options=model_combustible_data[modelo])
    transmision = st.selectbox('Transmisión', options=model_transmision_data[modelo])

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
    st.subheader('Características de Entrada:')
    st.write(input_data)

# Visualización de datos de entrada si está seleccionado
if show_data_info:
    st.subheader('Visualización de Datos de Entrada')

    # Histograma del año de los modelos con Plotly
    fig_hist = px.histogram(df, x='Año', nbins=30, title='Distribución de Años de los Modelos')
    st.plotly_chart(fig_hist)

    # Gráfico de dispersión del kilometraje vs precio con Plotly
    fig_scatter = px.scatter(df, x='Kilómetros', y='Precio', title='Relación entre Kilómetros y Precio')
    st.plotly_chart(fig_scatter)

    # Gráfico de depreciación del precio por año del modelo seleccionado
    df_modelo = df[(df['Marca'] == marca) & (df['Modelo'] == modelo)]
    precio_promedio_por_año = df_modelo.groupby('Año')['Precio'].mean().reset_index()
    fig_depreciacion = px.line(precio_promedio_por_año, x='Año', y='Precio', title=f'Depreciación del Precio de {marca} {modelo} por Año')
    st.plotly_chart(fig_depreciacion)

    # Gráfico de distribución de precios por año del modelo seleccionado
    fig_distribucion_precio = px.bar(df_modelo, x='Año', y='Precio', title=f'Distribución de Precios de {marca} {modelo} por Año')
    st.plotly_chart(fig_distribucion_precio)

# Botón para realizar la predicción
if st.button('Realizar Predicción'):
    with st.spinner('Realizando Predicción...'):
        # Transformar los datos de entrada
        input_data_transformado = input_to_features(input_data)

        # Realizar la predicción
        prediction = model.predict(input_data_transformado)

        # Mostrar el resultado de la predicción
        st.subheader('Resultado de la Predicción:')
        st.write(f'Usted puede vender su SUV por: ${prediction[0]:,.2f}')

# Explicación de la predicción si está seleccionado
if show_explanation:
    st.subheader('Explicación de la Predicción:')
    st.write('El modelo considera características como el año, kilometraje, tipo de motor, etc., para predecir el precio.')

# Personalización de tema
st.markdown(
    """
    <style>
    .fullScreenFrame {
        background-color: #f0f0f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)
