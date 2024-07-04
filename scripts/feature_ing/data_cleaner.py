import numpy as np
import pandas as pd
import time
import os
import json



class DataCleaner:
    # Clase DataCleaner
    # Esta clase se encarga de limpiar y preprocesar los datos del dataset de autos. Dependiendo si el dataset provisto es de entrenamiento o de evaluación, 
    # se aplican diferentes técnicas de limpieza y preprocesamiento.
    # Se eliminan outliers, se corrigen errores en los datos, se codifican las variables categóricas y se guardan los datos limpios en un nuevo archivo CSV.
    # Parámetros:
    # - data: DataFrame con los datos a limpiar
    # - price_usd: Precio del dólar en pesos argentinos
    # - eval_data: Booleano que indica si los datos son de evaluación o no
    
    
    def __init__(self, data, root, price_usd=950, eval_data=False):
        self.data = data
        self.eval_data = eval_data
        self.price_usd = price_usd
        self.model_encoding = None
        self.root = root
        if not self.eval_data: 
            self.clean_data()
        else:
            self.load_encodings()
            self.clean_eval_data()

        
    def clean_data(self):
        # clean_data
        # Función que limpia y preprocesa los datos del dataset de entrenamiento.
        # Corre en orden las funciones de limpieza y preprocesamiento de las variables del dataset.
        
        # ---- precio ----
        self.ars_to_usd()
        price_outliers = self.find_outliers('Precio', 400000)
        self.delete_rows(price_outliers)
        self.data = self.data.reset_index(drop=True)
        
        # ---- year ----
        self.clean_year()
        self.set_age()
        
        # ---- km ----
        self.km_to_int()
        avg_km = self.avg_km_year()
        same_digit_outliers = self.find_km_outliers_same_digit()
        real_km_outliers, saved_outliers = self.check_km_outliers(same_digit_outliers)
        self.rewrite_sample('Kilómetros', saved_outliers, 0)
        self.delete_rows(real_km_outliers)
        self.data = self.data.reset_index(drop=True)
        
        km_outliers = self.find_km_outliers(700000)
        self.delete_rows(km_outliers)
        self.data = self.data.reset_index(drop=True)
        
        # ---- engine ----
        self.process_engine()    
        self.correct_engines()
        
        # ---- cylinders ----
        self.process_cylinders()
        
        # ---- turbo ----
        self.process_turbo()
        
        # ---- color ----
        self.process_color()  
        self.encode_categorical('Color')  
        
        # ---- car manufacturer ----
        self.clean_car_manufacturer()
        
        # ---- one-hot encoding ----
        self.one_hot_encoding('Transmisión')
        self.one_hot_encoding('Marca')
        self.one_hot_encoding('Tipo de combustible')
        self.one_hot_encoding('Tipo de vendedor')
        
        # ---- target encoding ----
        self.model_encoding = self.target_encoding('Modelo', 'Precio')
        self.save_encodings()
        # ---- delete columns ----
           # ---- suv / cameras / doors / title ----
        self.delete_columns('Tipo de carrocería')
        self.delete_columns('Con cámara de retroceso')
        self.delete_columns('Puertas')
        self.delete_columns('Título')
        self.delete_columns('Versión')
        self.delete_columns('Moneda')
        
        # ----- save data -----
        self.rewrite_data('data/CLEAN_TRAIN_DATASET.csv')
        
        
        
        
        
    def clean_eval_data(self):
        # clean_eval_data
        # Función que limpia y preprocesa los datos del dataset de evaluación.
        # Corre en orden las funciones de limpieza y preprocesamiento de las variables del dataset.
        
        self.delete_columns('id')
        self.save_corrupt_data()
      
        # ---- year ----
        self.clean_year()
        self.set_age()
        
        # # ---- km ----
        self.km_to_int()
        avg_km = self.avg_km_year()
        same_digit_outliers = self.find_km_outliers_same_digit()
        real_km_outliers, saved_outliers = self.check_km_outliers(same_digit_outliers)
        self.rewrite_sample('Kilómetros', real_km_outliers, 10900, True)
        self.rewrite_sample('Kilómetros', saved_outliers, 0)
        
        km_outliers = self.find_km_outliers(700000)
        self.rewrite_sample('Kilómetros', km_outliers, 10900, True)
        
        # ---- engine ----
        self.process_engine()
        self.correct_engines()
        
        # ---- cylinders ----
        self.process_cylinders()
        
        # ---- turbo ----
        self.process_turbo()
        
        # ---- color ----
        self.process_color()
        self.encode_categorical('Color')
        
        # ---- car manufacturer ----
        self.clean_car_manufacturer()
        
        # ---- add columns ----
        self.add_new_feature(False, 'Marca_Abarth')
        self.add_new_feature(False, 'Marca_Sandero')
        self.add_new_feature(False, 'Tipo de combustible_Híbrido/Diesel')
        
        # ---- transmission ----
        automatico = self.data.loc[self.data['Transmisión'] == 'Automático'].index
        self.rewrite_sample('Transmisión', automatico, 'Automática')
        
        # ---- one-hot encoding ----
        self.one_hot_encoding('Transmisión')
        self.one_hot_encoding('Marca')
        self.one_hot_encoding('Tipo de combustible')
        self.one_hot_encoding('Tipo de vendedor')
        
        # ---- target encoding ----
        self.eval_model_encode()
        
        # ---- delete columns ----
            # ---- suv / cameras / doors / title ----
        self.delete_columns('Tipo de carrocería')
        self.delete_columns('Con cámara de retroceso')
        self.delete_columns('Puertas')
        self.delete_columns('Título')
        self.delete_columns('Versión')
        
        # ----- save data -----
        self.rewrite_data('data/CLEAN_TEST_DATASET.csv')
        
        
        
        
    ########### METODOS DE LIMPIEZA Y PREPROCESAMIENTO ###########
    
    # --------- Read and rewrite dataset ----------

    def rewrite_data(self, filename):
        # rewrite_data
        # Función que guarda los datos limpios en un nuevo archivo CSV.
        # Parámetros:
        # - filename: Ruta donde se guardará el archivo CSV con los datos limpios.
        
        data = self.data
        print("Saving data into file...")
        data.to_csv(os.path.join(self.root, filename), index=False)
        print("Data saved successfully!")

    def delete_rows(self, indexes):
        # delete_rows
        # Función que elimina las filas con los índices especificados.
        # Parámetros:
        # - indexes: Índices de las filas a eliminar.
        
        data = self.data
        indexes = indexes.sort_values(ascending=False)
        for index in indexes:
            data = data.drop(index)
        self.data = data
        
    def delete_columns(self, feature):
        # delete_columns
        # Función que elimina la columna especificada.
        # Parámetros:
        # - feature: Nombre de la columna a eliminar.
        
        data = self.data
        data = data.drop(feature, axis=1)
        self.data = data
        
    def rewrite_sample(self, feature, indexes, value, avg=False):
        # rewrite_sample
        # Función que reemplaza el valor de una cierta feature para un conjunto de índices.
        # Parámetros:
        # - feature: Nombre de la feature a modificar.
        # - indexes: Índices de las filas a modificar.
        # - value: Valor a reemplazar.
        
        data = self.data
        if avg == True:
            for index in indexes:
                data.loc[index, feature] = value * data.loc[index, 'Edad']
        for index in indexes:
            data.loc[index, feature] = value
        self.data = data


# --------- Clean price column ----------

    def ars_to_usd(self):
        # ars_to_usd
        # Función que convierte los precios de pesos argentinos a dólares.
        # Se utiliza el precio del dólar especificado en el constructor de la clase.
        
        data = self.data
        data.loc[data['Moneda'] == "$", 'Precio'] = round(data.loc[data['Moneda'] == "$", 'Precio'] / self.price_usd, 2)
        data.loc[data['Moneda'] == "$", 'Moneda'] = "U$S"
        self.data = data


# --------- Clean km column ---------- 
        
    def km_to_int (self):
        # km_to_int
        # Función que convierte la columna de kilómetros a enteros.
        
        data = self.data
        try:
            data['Kilómetros'] = data['Kilómetros'].str.replace(' km', '').astype(int)
        except:
            print("The column 'Kilómetros' is already clean")
        self.data = data
            
    def find_km_outliers_same_digit(self):
        # find_km_outliers_same_digit
        # Función que encuentra los outliers en la columna de kilómetros que tienen todos los dígitos iguales.
        # Retorna los índices de los outliers.
        
        data = self.data
        def is_same_digit(num):
            if num == 0:
                return False
            num_str = str(num)
            return all((ch == num_str[0]) for ch in num_str)
        
        same_digit_outliers = data.loc[data['Kilómetros'].apply(is_same_digit)].index
        return same_digit_outliers
    
    def find_km_outliers(self, threshold):
        # find_km_outliers
        # Función que encuentra los outliers en la columna de kilómetros que superan un cierto umbral.
        # Parámetros:
        # - threshold: Umbral para considerar un valor como outlier.
        # Retorna los índices de los outliers.
        
        data = self.data
        outliers = data.loc[data['Kilómetros'] >= threshold].index
        return outliers
    
    def check_km_outliers(self, p_outliers):
        # check_km_outliers
        # Función que verifica si los outliers en la columna de kilómetros son reales o no.
        # Se consideran outliers reales aquellos que sean anteriores a 2023.
        # Parámetros:
        # - p_outliers: Índices de los outliers a verificar.
        # Retorna los índices de los outliers reales y los outliers que que se pueden corregir.
        
        outliers = self.data.loc[p_outliers]
        saved_outliers = outliers.loc[(outliers['Año'] == 2023) | (outliers['Año'] == 2024)].index
        p_outliers = p_outliers.drop(saved_outliers)
        return p_outliers, saved_outliers

    def avg_km_year(self):
        # avg_km_year
        # Función que calcula el promedio de kilómetros recorridos por año.
        # Retorna el promedio de kilómetros por año.
        
        data = self.data
        data['Km promedio por año'] = data['Kilómetros'] // data['Edad'].replace(0, 1)
        self.data = data
        avg_km = data['Km promedio por año'].mean()
        return avg_km

# --------- Clean year column ----------

    def clean_year(self):
        # clean_year
        # Función que elimina los outliers en la columna de año. Considera outliers aquellos años posteriores al año actual + 1.
        # En el caso de los datos de entrenamiento se eliminan. Si los datos son de evaluación, 
        # se reemplazan los outliers por la cantidad de kilómetros dividido 10900, que es el promedio de kilómetros por año en los datos de entrenamiento.
        
        data = self.data
        outliers = data.loc[data['Año'] > (time.localtime().tm_year + 1)].index
        outliers = outliers.sort_values(ascending=False)
        if not outliers.empty:
            if not self.eval_data:
                self.delete_rows(outliers)
                self.data = self.data.reset_index(drop=True)
            else:
                self.rewrite_sample('Año', outliers, outliers['kilometros'] // 10900) # 10900 is the average km per year in the training data
        
    def set_age(self):
        # set_age
        # Función que calcula la edad de los autos y agrega una nueva columna al dataset.
        
        data = self.data
        current_year = time.localtime().tm_year
        data['Edad'] = current_year - data['Año']
        self.data = data


# --------- Generic functions ----------
    
    def find_outliers(self, feature, threshold):
        # find_outliers
        # Función que encuentra los outliers en una columna que superan un cierto umbral.
        # Parámetros:
        # - feature: Nombre de la columna a analizar.
        # - threshold: Umbral para considerar un valor como outlier.
        # Retorna los índices de los outliers.
        
        data = self.data
        outliers = data.loc[data[feature] >= threshold].index
        return outliers


    def convolucion_1d(self, feature, kernel):
        # convolucion_1d
        # Función que realiza una convolución 1D entre una feature y un kernel.
        # Parámetros:
        # - feature: Feature a analizar.
        # - kernel: Kernel a buscar en la feature.
        # Retorna True si el kernel está presente en la feature, False en caso contrario.
        
        if kernel.lower() in feature.lower():
            return True
        return False
    
# --------- Clean car manufacturer ----------

    def clean_car_manufacturer(self):
        # clean_car_manufacturer
        # Función que corrige errores en la columna de marca de los autos.
        self.data.loc[self.data['Marca'] == 'BMW x3 2.5 si xdrive', 'Marca'] = 'BMW'
        self.data.loc[self.data['Marca'] == 'Chrysler', 'Marca'] = 'Jeep'
        self.data.loc[self.data['Marca'] == 'DS7', 'Marca'] = 'DS'
        self.data.loc[self.data['Marca'] == 'DS AUTOMOBILES', 'Marca'] = 'DS'
        self.data.loc[self.data['Marca'] == 'hiunday', 'Marca'] = 'Hyundai'
        self.data.loc[self.data['Marca'] == 'Jetur', 'Marca'] = 'Jetour'

        

# --------- Clean color ----------


    def process_color(self):
        # process_color
        # Función que corrige errores en la columna de color de los autos.
        # Se mapean los colores a una lista de colores válidos meidante convoluciones 1D.
        # Se asignan colores a las muestras que no tienen un color valido mediante una distribución de probabilidad.
        
        
        data = self.data
        muestras_sin_color = []
        
        color_mappings = {
            'blanc': 'blanco',
            'negr': 'negro',
            'plat': 'plata', 'silver': 'plata',
            'gr': 'gris',
            'rojo': 'rojo',
            'bordó': 'bordó',
            'verde': 'verde',
            'azul': 'azul', 'blue': 'azul',
            'a ele': 'elección', 'todos': 'elección',
            'rosa': 'rosa',
            'violeta': 'violeta',
            'celeste': 'celeste',
            'dorado': 'dorado',
            'amarillo': 'amarillo',
            'naranja': 'naranja', 'orange': 'naranja',
            'marrón': 'marrón', 'ocre': 'marrón', 'beige': 'marrón'
        }
        
        for i in range(len(data)):
            color = str(data.loc[i, 'Color']).lower()
            found_color = False
            
            for pattern, new_color in color_mappings.items():
                if self.convolucion_1d(color, pattern):
                    data.loc[i, 'Color'] = new_color
                    found_color = True
                    break
            
            if not found_color:
                muestras_sin_color.append(i)

        color_distribution = data.loc[~data.index.isin(muestras_sin_color), 'Color'].value_counts(normalize=True)

        colors = color_distribution.index.tolist()
        probabilities = color_distribution.values

        data.loc[muestras_sin_color, 'Color'] = np.random.choice(colors, size=len(muestras_sin_color), p=probabilities)

        self.data = data


## --------- Process engine spects ----------

    def process_cylinders(self):
        # process_cylinders
        # Función que determina la cantidad de cilindros de los autos a partir de la versión.
        # Se utilizan convoluciones 1D para buscar palabras clave en la versión que indiquen la cantidad de cilindros.
        
        data = self.data
        data['Cilindros'] = data['Versión'].apply(lambda x: 2 if self.convolucion_1d(str(x), 'v6') else (3 if self.convolucion_1d(str(x), 'v8') else 1))
        self.data = data

    def process_turbo(self):
        # process_turbo
        # Función que determina si los autos tienen turbo a partir de la versión.
        # Se utilizan convoluciones 1D para buscar palabras clave en la versión que indiquen la presencia de un turbo.
        # Se agrega una nueva columna al dataset con valores binarios.
        
        data = self.data
        data['Turbo'] = data['Versión'].apply(lambda x: 1 if self.convolucion_1d(str(x), 'turbo') else 0)
        self.data = data


    def process_engine(self):  
        # process_engine
        # Función que corrige los valores de la columna de motor de los autos.
        # Se extraen los valores numéricos de la columna y se buscan valores válidos mediante convoluciones 1D.
        # Los autos híbridos y eléctricos se clasifican como 'No aplica'.
        # Los autos que no tienen especificado el motor se clasifican como 'No especificado'.
        
        data = self.data
        
        motor_values = data['Motor'].str.extract(r'(\d+\.\d+)').dropna().astype(float)
        max_engine = motor_values.max().iloc[0]
        min_engine = motor_values.min().iloc[0]

        kernels = [] 
        linespace = np.arange(min_engine, max_engine+0.1, 0.1)
        for i in linespace:
            kernels.append(str(round(i, 1)))

        for j in range(len(data)):
            motor = str(data.loc[j, 'Motor'])  
            version = str(data.loc[j, 'Versión'])

            found = False
            for kernel in kernels:
                if self.convolucion_1d(motor, kernel) or self.convolucion_1d(version, kernel):
                    data.loc[j, 'Motor'] = kernel
                    found = True
                    break

            if not found:
                if str(data.loc[j, 'Tipo de combustible']) not in ['Híbrido', 'Eléctrico']:
                    data.loc[j, 'Motor'] = 'No especificado'
                else:
                    data.loc[j, 'Motor'] = 'No aplica'  

        self.data = data
        
    def correct_engines(self):
        # correct_engines
        # Función que corrige los valores de la columna de motor de los autos.
        # Los autos electrícos se clasifican como '-1'.
        # Se buscan valores en autos con el mismo modelo o marca y se reemplazan los valores incorrectos por valores válidos.
        
        data = self.data
        wrong_engines = data.loc[(data['Motor'] == 'No aplica') | (data['Motor'] == 'No especificado')].index
 
        for index in wrong_engines:
            model = data.loc[index, 'Modelo']
            valid_engines = data.loc[(data['Modelo'] == model) & ~(data['Motor'].isin(['No aplica', 'No especificado']))]['Motor']
            
            if not valid_engines.empty:
                data.loc[index, 'Motor'] = valid_engines.iloc[0]
            elif data.loc[index, 'Motor'] == 'No aplica':
                if data.loc[index, 'Tipo de combustible'] == 'Híbrido':
                    data.loc[index, 'Motor'] = '1.5'
                elif data.loc[index, 'Tipo de combustible'] == 'Eléctrico':
                    data.loc[index, 'Motor'] = '-1'
            else:
                marca = data.loc[index, 'Marca']
                valid_engines = data.loc[(data['Marca'] == marca) & ~(data['Motor'].isin(['No especificado']))]
                if self.eval_data:
                    avg_engine = valid_engines['Motor'].astype(float).mean()
                    data.loc[index, 'Motor'] = avg_engine
                else:
                    if not valid_engines.empty:
                        prices = valid_engines['Precio']
                        closest_price = min(prices, key=lambda x: abs(x - data.loc[index, 'Precio']))
                        data.loc[index, 'Motor'] = data.loc[data['Precio'] == closest_price, 'Motor'].iloc[0]

                    else:
                        data.loc[index, 'Motor'] = 'No especificado'
    
        self.data = data


# --------- Encoding categorical features ----------

    def encode_categorical(self, feature):
        # encode_categorical
        # Función que codifica las variables categóricas en el dataset.
        # Parámetros:
        # - feature: Nombre de la columna a codificar.
        
        data = self.data
        data[feature] = data[feature].astype('category')
        data[feature] = data[feature].cat.codes

        self.data = data


    def one_hot_encoding(self, feature):
        # one_hot_encoding
        # Función que realiza one-hot encoding en una variable categórica.
        # Parámetros:
        # - feature: Nombre de la columna a codificar.
        
        data = self.data
        data = pd.get_dummies(data, columns=[feature])
        self.data = data
        
    def add_new_feature(self, feature, new_feature):
        # add_new_feature
        # Función que agrega una nueva característica al dataset.
        # Parámetros:
        # - feature: Nombre de la columna a utilizar para crear la nueva característica.
        # - new_feature: Nombre de la nueva característica a agregar.
        
        data = self.data
        data[new_feature] = feature
        self.data = data

    # def target_encoding(self, feature, target):
    #     # target_encoding
    #     # Función que realiza target encoding en una variable categórica.
    #     # Parámetros:
    #     # - feature: Nombre de la columna a codificar.
    #     # - target: Nombre de la columna target.
        
    #     data = self.data
    #     data[feature] = data.groupby(feature)[target].transform('mean').astype(int)
    #     self.data = data
        
    
    def target_encoding(self, feature, target):
        # target_encoding
        # Función que realiza target encoding en una variable categórica.
        # Parámetros:
        # - feature: Nombre de la columna a codificar.
        # - target: Nombre de la columna target.
        
        data = self.data
        encoding_dict = data.groupby(feature)[target].mean().astype(int).to_dict()
        data[feature] = data[feature].map(encoding_dict)
        self.data = data
        return encoding_dict
    
    def eval_model_encode(self):
        data = self.data
        encoding_dict = self.model_encoding
        for index in range(len(data)):
            try:
                data.loc[index, 'Modelo'] = encoding_dict[data.loc[index, 'Modelo']]
            except:
                print(f"Error in sample {index} with model {data.loc[index, 'Modelo']}")
                
                data.loc[index, 'Modelo'] = 0
            
    # data['Modelo'] = data['Modelo'].map(encoding_dict)
    
    # ------- SAVE DATA -------
    def save_corrupt_data(self):
        data = self.data
        corrupt_samples = data.loc[data.isnull().sum(axis=1) > 3].index
        for sample in corrupt_samples:
            data.loc[sample] = data.loc[sample+1]
        self.data = data
        
    def save_encodings(self):
        with open(os.path.join(self.root, 'scripts/feature_ing/model_encoding.json'), 'w') as file:
            json.dump(self.model_encoding, file)
        print("Encodings saved successfully!")
        
    def load_encodings(self):
        with open(os.path.join(self.root, 'scripts/feature_ing/model_encoding.json'), 'r') as file:
            self.model_encoding = json.load(file)
        print("Encodings loaded successfully!")
        