import streamlit as st
import pandas as pd
import os
import sys
from joblib import load
import plotly.express as px
import numpy as np
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

# Crear un diccionario con los modelos que tienen turbo
model_turbo_data = {}
for modelo in df['Modelo'].unique():
    model_turbo_data[modelo] = df[df['Modelo'] == modelo]['Turbo'].unique()



# Crear la interfaz de Streamlit
st.set_page_config(page_title='Predicción de Precios de SUVs', page_icon='🚙')


# Añadir un logo
logo_path = os.path.join(project_root, 'TPFinal-ML/extra/logo.png')
if os.path.exists(logo_path):
    st.image(logo_path, use_column_width=True)

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
    año = st.selectbox('Año', options=sorted(model_year_data[modelo]))
    color = st.selectbox('Color', options=model_color_data[modelo])
    combustible = st.selectbox('Combustible', options=model_combustible_data[modelo])
    transmision = st.selectbox('Transmisión', options=model_transmision_data[modelo])
    if model_turbo_data[modelo][0] == 1:
        turbo = st.selectbox('Turbo', options=['Si', 'No'])
    else:
        turbo = 'No'

    # Crear un diccionario con las características de entrada
    input_features = {
        'Marca': marca,
        'Modelo': modelo,
        'Motor': motor,
        'Kilómetros': kilometraje,
        'Año': año,
        'Color': color,
        'Tipo de combustible': combustible,
        'Transmisión': transmision,
        'Turbo': turbo
    }

    # Convertir las características de entrada a un DataFrame
    input_data = pd.DataFrame([input_features])

    # Mostrar las características de entrada
    st.subheader('Características de Entrada:')
    st.write(input_data)

# Visualización de datos de entrada si está seleccionado
if show_data_info:
    st.subheader('Visualización de Datos de Entrada')

    # Gráfico de depreciación del precio por año del modelo seleccionado
    df_modelo = df[(df['Marca'] == marca) & (df['Modelo'] == modelo)]
 
    año_entrada = input_data['Año'][0]
    # Gráfico de depreciación del precio futuro
    años_futuros = np.arange(2024, 2030)
    años_deprecacion = np.arange(año_entrada, año_entrada - 6, -1)
    predicciones_futuras = []
    for año in años_deprecacion:
        input_data_futuro = input_data.copy()
        input_data_futuro['Año'] = año 
        input_data_futuro['Kilómetros'] = kilometraje + np.abs(año - 2024) * 10000
        input_data_transformado_futuro = input_to_features(input_data_futuro)
        prediccion_futura = model.predict(input_data_transformado_futuro)
        predicciones_futuras.append(prediccion_futura[0])
    
    df_predicciones_futuras = pd.DataFrame({'Año': años_futuros, 'Precio': predicciones_futuras})
    fig_depreciacion_futura = px.line(df_predicciones_futuras, x='Año', y='Precio', title=f'Depreciación Futura del Precio de {marca} {modelo}')
    st.plotly_chart(fig_depreciacion_futura)
    
    # Gráfico de valor relativo de los colores del modelo seleccionado
    predicciones_por_motor = []
    for motor in sorted(model_motor_data[modelo]):
        input_data_motor = input_data.copy()
        input_data_motor['Motor'] = motor
        input_data_transformado_motor = input_to_features(input_data_motor)
        prediccion_motor = model.predict(input_data_transformado_motor)
        predicciones_por_motor.append({'Motor': motor, 'Precio': prediccion_motor[0]})
        
    df_predicciones_por_motor = pd.DataFrame(predicciones_por_motor)
    fig_valor_por_motor = px.bar(df_predicciones_por_motor, x='Motor', y='Precio', title=f'Valor Relativo por Motor de {marca} {modelo}', text='Precio')
    fig_valor_por_motor.update_traces(texttemplate='%{text:.2s}', textposition='outside', marker_line_width=0)
    fig_valor_por_motor.update_layout(xaxis_type='category')
    st.plotly_chart(fig_valor_por_motor)


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
        
    # Información sobre el modelo XGBoost
    st.write("**Modelo XGBoost:**")
    st.write("""
    XGBoost es un potente algoritmo de aprendizaje automático basado en árboles de decisión, conocido por su eficiencia y precisión. Es considerado de los mejores algoritmos para manejo de datos tabulares. Algunas características clave de XGBoost incluyen:
    - **Boosting**: XGBoost utiliza un proceso de boosting, que combina varios árboles de decisión débiles para crear un modelo fuerte. Cada nuevo árbol corrige los errores de los árboles anteriores.
    - **Regularización**: Incluye parámetros de regularización para prevenir el sobreajuste, lo que permite que el modelo generalice mejor en datos no vistos.
    - **Manejo de datos faltantes**: XGBoost tiene la capacidad de manejar datos faltantes de manera efectiva, identificando los mejores valores de división.
    - **Optimización de parámetros**: Se ha realizado una exhaustiva optimización de hiperparámetros utilizando una técnica de búsqueda que se basa en un enfoque probabilístico para encontrar la mejor combinación de parámetros. Este proceso es conocido como optimización bayesiana, el cual permite encontrar la mejor combinación de hiperparámetros en menos iteraciones.
    """)

    # Información adicional sobre la predicción
    st.write("**Cómo se realiza la predicción:**")
    st.write("""
    Para realizar la predicción, el modelo toma las características seleccionadas por el usuario, las transforma según las técnicas de preprocesamiento aplicadas (como la codificación y la limpieza de las muestras), y luego utiliza estas características para calcular el precio estimado del SUV. La predicción final es el resultado del modelo XGBoost que ha sido entrenado y optimizado en el conjunto de datos de entrenamiento.
    """)

    # Información sobre las características utilizadas por el modelo
    st.write("**Características utilizadas por el modelo:**")
    st.write("""
    Nuestro modelo XGBoost utiliza varias características del SUV para predecir su precio. Estas características incluyen:
    - **Marca**: La marca del vehículo, que puede influir en su valor de mercado debido a la reputación y fiabilidad percibida.
    - **Modelo**: El modelo específico, que tiene sus propias características y niveles de demanda.
    - **Motor**: Tipo de motor, que afecta el rendimiento y la eficiencia del vehículo.
    - **Kilómetros**: El kilometraje del vehículo, que es un indicador clave del desgaste y la vida útil restante.
    - **Año**: El año de fabricación, que impacta la depreciación del valor del vehículo.
    - **Color**: El color del vehículo, que puede influir en la preferencia del comprador y, por lo tanto, en el precio.
    - **Tipo de combustible**: Tipo de combustible utilizado, que puede afectar tanto los costos operativos como las preferencias de los consumidores.
    - **Transmisión**: Tipo de transmisión (manual o automática), que también puede influir en las preferencias de los compradores.
    - **Características del motor**: Características específicas del motor, como la distribución de cilindros y la alimentación de aire, que pueden influir en el rendimiento y la eficiencia.         
    """)



