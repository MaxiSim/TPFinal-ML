import numpy as np
import pandas as pd
import random
import time

class DataCleaner:
    
    def __init__(self, data, price_usd=100):
        self.data = data
        self.price_usd = price_usd
        self.clean_data
        
    def clean_data(self):
        # ---- precio ----
        self.ars_to_usd()
        price_outliers = self.find_outliers('Precio', 400000)
        self.delete_rows(price_outliers)
    
    
    # --------- Read and rewrite dataset ----------

    def rewrite_data(self, data, filename):
        data.to_csv(filename, index=False)

    def delete_rows(self, indexes):
        data = self.data
        indexes = indexes.sort_values(ascending=False)
        for index in indexes:
            data = data.drop(index)
        self.data = data
        
    def delete_columns(self, feature):
        data = self.data
        data = data.drop(feature, axis=1)
        self.data = data


# --------- Clean price column ----------

    def ars_to_usd(self):
        data = self.data
        data.loc[data['Moneda'] == "$", 'Precio'] = round(data.loc[data['Moneda'] == "$", 'Precio'] / self.price_usd, 2)
        data.loc[data['Moneda'] == "$", 'Moneda'] = "U$S"
        self.data = data


# --------- Clean km column ---------- 
        
    def km_to_int (self, data, filename):
        try:
            data['Kilómetros'] = data['Kilómetros'].str.replace(' km', '').astype(int)
        except:
            print("The column 'Kilómetros' is already clean")
        self.rewrite_data(data, filename)
        
    def find_km_outliers(self, data, threshold):
        def is_same_digit(num):
            if num == 0:
                return False
            num_str = str(num)
            return all((ch == num_str[0]) for ch in num_str)
        same_digit_outliers = data.loc[data['Kilómetros'].apply(is_same_digit)].index
        outliers = data.loc[data['Kilómetros'] >= threshold].index
        return outliers, same_digit_outliers

    def avg_km_year(self, data):
        current_year = time.localtime().tm_year
        data['Edad'] = current_year - data['Año']
        data['Km promedio por año'] = data['Kilómetros'] // data['Edad'].replace(0, 1)
        self.data = data

# --------- Clean year column ----------

    def clean_year(self, data, filename):
        outliers = data.loc[data['Año'] > time.localtime().tm_year].index
        outliers = outliers.sort_values(ascending=False)
        self.delete_rows(data, outliers, filename)

# --------- Clean door column ----------

    def clean_doors(self, data, filename):
        outliers = data.loc[data['Puertas'] > 5 or data['Puertas'] < 4].index
        outliers = outliers.sort_values(ascending=False)
        self.delete_rows(data, outliers, filename)


# --------- Generic functions ----------
    
    def find_outliers(self, feature, threshold):
        data = self.data
        outliers = data.loc[data[feature] >= threshold].index
        return outliers


    def convolucion_1d(self, feature, kernel):
        if kernel.lower() in feature.lower():
            return True
        return False
        

## --------- Clean color ----------


    def process_color(self):
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
        data = data.drop(muestras_sin_color)              ######### PREGUNTAR A MAXI ##########

        self.data = data

    def process_color_indef(self):
        data = self.data
        
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
        
        muestras_sin_color = []
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

        data.loc[data['Color'].isna(), 'Color'] = 'indefinido'
        self.data = data


## --------- Process engine spects ----------

    def process_cylinders(self):
        data = self.data
        data['Cilindros'] = data['Versión'].apply(lambda x: 2 if self.convolucion_1d(str(x), 'v6') else (3 if convolucion_1d(str(x), 'v8') else 1))
        self.data = data


    def process_turbo(self):
        data = self.data
        data['Turbo'] = data['Versión'].apply(lambda x: 1 if self.convolucion_1d(str(x), 'turbo') else 0)
        self.data = data


    def process_engine(self):
        data = self.data

        max_engine = float(data['Motor'].str.extract(r'(\d+\.\d+)').dropna().max())
        min_engine = float(data['Motor'].str.extract(r'(\d+\.\d+)').dropna().min())

        kernels = [] 
        linespace = np.arange(min_engine, max_engine+0.1, 0.1)
        for i in linespace:
            kernels.append(str(round(i, 1)))

        for j in range(len(data)):
            motor = str(data.loc[j, 'Motor'])
            version = str(data.loc[j, 'Versión'])

            found = False
            for kernel in kernels:
                if convolucion_1d(motor, kernel) or convolucion_1d(version, kernel):
                    data.loc[j, 'Motor'] = kernel
                    found = True
                    break

            if not found:
                if str(data.loc[j, 'Tipo de combustible']) not in ['Híbrido', 'Eléctrico']:
                    data.loc[j, 'Motor'] = 'No especificado'
                else:
                    data.loc[j, 'Motor'] = 'No aplica'

        self.data = data


    # def process_version(filename):
        
    #     data = read_data(filename)

    #     marca = random.choice(data['Marca'].unique())
    #     modelo = random.choice(data.loc[data['Marca'] == marca]['Modelo'].unique())
    #     # anio = random.choice(data.loc[(data['Marca'] == marca) & (data['Modelo'] == modelo)]['Año'].unique())
    #     versiones = data.loc[(data['Marca'] == marca) & (data['Modelo'] == modelo)]['Versión']
    #     print(f'Versiones del modelo {modelo}: {versiones.unique()}')    
    #     precios_por_cada_version = []

    #     for version in versiones:
    #         precios = data.loc[(data['Marca'] == marca) & (data['Modelo'] == modelo) & (data['Versión'] == version)]['Precio'].values
    #         precios_por_cada_version.append(precios)
    #         print(f'Precios de la versión {version}: {precios}')
            
    #     fig = go.Figure()
    #     for i in range(len(versiones)):
    #         fig.add_trace(go.Scatter(x=data.loc[(data['Marca'] == marca) & (data['Modelo'] == modelo) & (data['Versión'] == versiones.iloc[i])]['Año'], y=precios_por_cada_version[i], mode='markers', name=versiones.iloc[i]))
    #     fig.update_layout(title_text=f'Prices of versions of model {modelo} ', xaxis_title_text='Year', yaxis_title_text='Price')
    #     fig.show()    



# --------- Encoding categorical features ----------

    def encode_categorical(self, data, feature):
        data[feature] = data[feature].astype('category')
        data[feature] = data[feature].cat.codes

        self.data = data


    def one_hot_encoding(self, data, feature):
        data = pd.get_dummies(data, columns=[feature])
        self.data = data

    def target_encoding(self, data, feature, target):
        data[feature] = data.groupby(feature)[target].transform('mean').astype(int)
        self.data = data