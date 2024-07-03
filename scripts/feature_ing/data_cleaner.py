import numpy as np
import pandas as pd
import random
import time
from filtrado import read_data

class DataCleaner:
    
    def __init__(self, data, price_usd=950, eval_data=False):
        self.data = data
        self.eval_data = eval_data
        self.price_usd = price_usd
        if not self.eval_data:
            self.clean_data()
        else:
            self.clean_eval_data()

        
    def clean_data(self):
        # ---- precio ----
        self.ars_to_usd()
        price_outliers = self.find_outliers('Precio', 400000)
        self.delete_rows(price_outliers)
        self.data = self.data.reset_index(drop=True)
        
        # ---- year ----
        self.clean_year()        ###### esto borra filas
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
        
        # ---- one-hot encoding ----
        self.one_hot_encoding('Transmisión')
        self.one_hot_encoding('Marca')
        self.one_hot_encoding('Tipo de combustible')
        self.one_hot_encoding('Tipo de vendedor')
        
        # ---- target encoding ----
        self.target_encoding('Modelo', 'Precio')
        
        # ---- delete columns ----
           # ---- suv / cameras / doors / title ----
        self.delete_columns('Tipo de carrocería')
        self.delete_columns('Con cámara de retroceso')
        self.delete_columns('Puertas')
        self.delete_columns('Título')
        self.delete_columns('Versión')
        self.delete_columns('Moneda')
        
        
        # ----- save data -----
        self.rewrite_data('../../src/FINAL_DATASET.csv')
        
        
        
        
        
    def clean_eval_data(self):
        # ---- precio ----
        self.ars_to_usd()
      
        # ---- year ----
        self.clean_year()
        self.set_age()
        
        # ---- km ----
        self.km_to_int()
        avg_km = self.avg_km_year(self.data)
        km_outliers, same_digit_outliers = self.find_km_outliers(700000)
        real_km_outliers, saved_outliers = self.check_km_outliers(same_digit_outliers)
        self.rewrite_sample('Kilómetros', real_km_outliers, avg_km * self.data['Edad'])
        self.rewrite_sample('Kilómetros', km_outliers, avg_km * self.data['Edad'])
        self.rewrite_sample('Kilómetros', saved_outliers, 0)
        
        # ---- engine ----
        self.process_engine()
        
        # ---- cylinders ----
        self.process_cylinders()
        
        # ---- turbo ----
        self.process_turbo()
        
        # ---- color ----
        self.process_color()
        
        # ---- encode categorical features ----
        self.encode_categorical('Color')
        
        # ---- one-hot encoding ----
        self.one_hot_encoding('Transmisión')
        self.one_hot_encoding('Marca')
        self.one_hot_encoding('Tipo de combustible')
        self.one_hot_encoding('Tipo de vendedor')
        
        # ---- target encoding ----
        self.target_encoding('Modelo', 'Precio')
        
        # ---- delete columns ----
            # ---- suv / cameras / doors / title ----
        self.delete_columns('Tipo de carrocería')
        self.delete_columns('Con cámara de retroceso')
        self.delete_columns('Puertas')
        self.delete_columns('Título')
        self.delete_columns('Versión')
        self.delete_columns('Moneda')
        
        # ----- save data -----
        self.rewrite_data('../../src/FINAL_EVAL_DATASET.csv')
    
    # --------- Read and rewrite dataset ----------

    def rewrite_data(self, filename):
        data = self.data
        print("Saving data into file...")
        data.to_csv(filename, index=False)
        print("Data saved successfully!")

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
        
    def rewrite_sample(self, feature, indexes, value):
        data = self.data
        for index in indexes:
            data.loc[index, feature] = value
        self.data = data


# --------- Clean price column ----------

    def ars_to_usd(self):
        data = self.data
        data.loc[data['Moneda'] == "$", 'Precio'] = round(data.loc[data['Moneda'] == "$", 'Precio'] / self.price_usd, 2)
        data.loc[data['Moneda'] == "$", 'Moneda'] = "U$S"
        self.data = data


# --------- Clean km column ---------- 
        
    def km_to_int (self):
        data = self.data
        try:
            data['Kilómetros'] = data['Kilómetros'].str.replace(' km', '').astype(int)
        except:
            print("The column 'Kilómetros' is already clean")
        self.data = data
        
    def find_km_outliers_same_digit(self):
        data = self.data
        def is_same_digit(num):
            if num == 0:
                return False
            num_str = str(num)
            return all((ch == num_str[0]) for ch in num_str)
        
        same_digit_outliers = data.loc[data['Kilómetros'].apply(is_same_digit)].index
        return same_digit_outliers
    
    def find_km_outliers(self, threshold):
        data = self.data
        outliers = data.loc[data['Kilómetros'] >= threshold].index
        return outliers
    
    def check_km_outliers(self, p_outliers):
        outliers = self.data.loc[p_outliers]
        saved_outliers = outliers.loc[(outliers['Año'] == 2023) | (outliers['Año'] == 2024)].index
        p_outliers = p_outliers.drop(saved_outliers)
        return p_outliers, saved_outliers

    def avg_km_year(self):
        data = self.data
        data['Km promedio por año'] = data['Kilómetros'] // data['Edad'].replace(0, 1)
        self.data = data
        avg_km = data['Km promedio por año'].mean()
        return avg_km

# --------- Clean year column ----------

    def clean_year(self):
        data = self.data
        outliers = data.loc[data['Año'] > (time.localtime().tm_year + 1)].index
        outliers = outliers.sort_values(ascending=False)
        if not self.eval_data:
            self.delete_rows(outliers)
            self.data = self.data.reset_index(drop=True)
        else:
            self.rewrite_sample('Año', outliers, outliers['kilometros'] // 10900) # 10900 is the average km per year in the training data
        
    def set_age(self):
        data = self.data
        current_year = time.localtime().tm_year
        data['Edad'] = current_year - data['Año']
        self.data = data

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
        data['Cilindros'] = data['Versión'].apply(lambda x: 2 if self.convolucion_1d(str(x), 'v6') else (3 if self.convolucion_1d(str(x), 'v8') else 1))
        self.data = data

    def process_turbo(self):
        data = self.data
        data['Turbo'] = data['Versión'].apply(lambda x: 1 if self.convolucion_1d(str(x), 'turbo') else 0)
        self.data = data


    def process_engine(self):  
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
                if not valid_engines.empty:
                    prices = valid_engines['Precio']
                    closest_price = min(prices, key=lambda x: abs(x - data.loc[index, 'Precio']))
                    data.loc[index, 'Motor'] = data.loc[data['Precio'] == closest_price, 'Motor'].iloc[0]

                else:
                    data.loc[index, 'Motor'] = 'No especificado'
    
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

    def encode_categorical(self, feature):
        data = self.data
        data[feature] = data[feature].astype('category')
        data[feature] = data[feature].cat.codes

        self.data = data


    def one_hot_encoding(self, feature):
        data = self.data
        data = pd.get_dummies(data, columns=[feature])
        self.data = data

    def target_encoding(self, feature, target):
        data = self.data
        data[feature] = data.groupby(feature)[target].transform('mean').astype(int)
        self.data = data