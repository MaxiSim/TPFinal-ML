import json


### 

# JSON con los valores asociados a cada modelo
model_value_json = '''
{
    "2008": 16077,
    "208": 23745,
    "3008": 44159,
    "4008": 15248,
    "4Runner": 15339,
    "5008": 47160,
    "500X": 19705,
    "7": 73333,
    "Actyon": 10000,
    "Agile": 7367,
    "Aircross": 20347,
    "Amigo": 14500,
    "Blazer": 7894,
    "Bronco": 48677,
    "Bronco Sport": 45101,
    "C-HR": 48000,
    "C3": 19466,
    "C3 Aircross": 22940,
    "C4": 23894,
    "C4 Aircross": 15413,
    "C4 Cactus": 20627,
    "C5 Aircross": 42422,
    "CR-V": 16197,
    "Captiva": 14461,
    "Captur": 18713,
    "Cayenne": 67480,
    "Cherokee": 19432,
    "Clase E": 295000,
    "Clase GL": 57900,
    "Clase GLA": 43059,
    "Clase GLB": 68377,
    "Clase GLC": 62563,
    "Clase GLE": 122276,
    "Clase GLK": 21277,
    "Clase ML": 32066,
    "Commander": 57340,
    "Compass": 33718,
    "Cooper Countryman": 45814,
    "Corolla Cross": 34381,
    "Countryman": 93900,
    "Coupe": 63900,
    "Creta": 25247,
    "Crossfox": 7723,
    "DS3": 53845,
    "DS7": 69200,
    "DS7 Crossback": 55818,
    "Defender": 63933,
    "Discovery": 25602,
    "Duster": 15188,
    "Duster Oroch": 17143,
    "E-tron": 166200,
    "Ecosport": 12550,
    "Emgrand X7 Sport": 18436,
    "Equinox": 39605,
    "Evoque": 46684,
    "Explorer": 8480,
    "F-PACE": 65500,
    "Feroza": 12017,
    "Forester": 23162,
    "Freelander": 11385,
    "Galloper": 7571,
    "Grand Blazer": 11052,
    "Grand Cherokee": 41188,
    "Grand Santa Fé": 33849,
    "Grand Vitara": 11019,
    "H1": 17252,
    "H6": 38673,
    "HR-V": 21982,
    "Hilux": 31900,
    "Hilux SW4": 31227,
    "Jimny": 15550,
    "Jolion": 32605,
    "Journey": 13779,
    "Kangoo": 24905,
    "Kicks": 23633,
    "Koleos": 20719,
    "Kona": 28850,
    "Kuga": 23682,
    "LX": 159000,
    "Land Cruiser": 41844,
    "ML": 24795,
    "Macan": 132776,
    "Mohave": 15611,
    "Montero": 13856,
    "Murano": 19442,
    "Musso": 6881,
    "Mustang": 113865,
    "Myway": 13067,
    "NX": 66813,
    "Nativa": 9960,
    "Nivus": 24546,
    "Oroch": 26965,
    "Outback": 16053,
    "Outlander": 13193,
    "Panamera": 131333,
    "Pathfinder": 10164,
    "Patriot": 13007,
    "Pilot": 31840,
    "Pulse": 18541,
    "Q2": 44900,
    "Q3": 35266,
    "Q3 Sportback": 63218,
    "Q5": 44996,
    "Q7": 47241,
    "Q8": 162116,
    "RAV4": 26544,
    "Range Rover": 31930,
    "Range Rover Sport": 44200,
    "Renegade": 24189,
    "Rodeo": 8620,
    "S2": 19234,
    "S5": 18522,
    "SQ5": 76631,
    "SW4": 39878,
    "Samurai": 11999,
    "Sandero": 13569,
    "Sandero Stepway": 16139,
    "Santa Fe": 17395,
    "Seltos": 31883,
    "Serie 4": 55588,
    "Sorento": 21292,
    "Soul": 12560,
    "Spin": 19193,
    "Sportage": 27843,
    "Stelvio": 78315,
    "Suran": 12838,
    "T-Cross": 21957,
    "Taos": 36012,
    "Terios": 6156,
    "Terrano II": 7642,
    "Territory": 37285,
    "Tiggo": 9032,
    "Tiggo 2": 15563,
    "Tiggo 3": 13218,
    "Tiggo 4": 24431,
    "Tiggo 4 Pro": 29214,
    "Tiggo 5": 15823,
    "Tiggo 8 Pro": 49981,
    "Tiguan": 19639,
    "Tiguan Allspace": 42114,
    "Touareg": 25102,
    "Tracker": 21433,
    "Trailblazer": 38220,
    "Trooper": 11058,
    "Tucson": 17677,
    "UX": 45276,
    "Veracruz": 14121,
    "Vitara": 12237,
    "Wrangler": 48779,
    "X-Terra": 12473,
    "X-Trail": 36274,
    "X1": 35049,
    "X2": 62913,
    "X25": 13236,
    "X3": 46465,
    "X35": 24892,
    "X4": 50878,
    "X5": 53049,
    "X55": 37050,
    "X6": 59690,
    "X70": 32321,
    "XC40": 64192,
    "XC60": 35661,
    "q5 sportback": 112660
}
'''


def get_model_value(model):
    """
    Función que devuelve el valor asociado a un modelo dado.
    
    Args:
    - model (str): Nombre del modelo de automóvil.
    
    Returns:
    - int: Valor asociado al modelo. Retorna 0 si el modelo no está en el diccionario.
    """
    # Cargar el JSON como diccionario
    model_value_dict = json.loads(model_value_json)
    return model_value_dict.get(model, 0)


def input_to_features(input_data):
    # Crear un diccionario con las características de entrada
    # mi data debe tener las mismas columnas que el modelo para que pueda predecir
    # Las columnas con las que entrenó el modelo son:
    # [['Modelo', 'Año', 'Color', 'Motor', 'Kilómetros', 'Edad', 'Km promedio por año', 'Cilindros', 'Turbo', 
    # 'Transmisión_Automática', 'Transmisión_Automática secuencial', 'Transmisión_Manual', 'Transmisión_Semiautomática', 
    # 'Marca_Abarth', 'Marca_Alfa Romeo', 'Marca_Audi', 'Marca_BAIC', 'Marca_BMW', 'Marca_Chery', 'Marca_Chevrolet', 
    # 'Marca_Citroën', 'Marca_DS', 'Marca_Daihatsu', 'Marca_Dodge', 'Marca_Fiat', 'Marca_Ford', 'Marca_Geely', 
    # 'Marca_Haval', 'Marca_Honda', 'Marca_Hyundai', 'Marca_Isuzu', 'Marca_JAC', 'Marca_Jaguar', 'Marca_Jeep', 
    # 'Marca_Jetour', 'Marca_Kia', 'Marca_Land Rover', 'Marca_Lexus', 'Marca_Lifan', 'Marca_MINI', 'Marca_Mercedes-Benz', 
    # 'Marca_Mitsubishi', 'Marca_Nissan', 'Marca_Peugeot', 'Marca_Porsche', 'Marca_Renault', 'Marca_Sandero', 
    # 'Marca_Ssangyong', 'Marca_Subaru', 'Marca_Suzuki', 'Marca_Toyota', 'Marca_Volkswagen', 'Marca_Volvo', 
    # 'Tipo de combustible_Diésel', 'Tipo de combustible_Eléctrico', 'Tipo de combustible_GNC', 'Tipo de combustible_Híbrido', 
    # 'Tipo de combustible_Híbrido/Diesel', 'Tipo de combustible_Híbrido/Nafta', 'Tipo de combustible_Nafta', 
    # 'Tipo de combustible_Nafta/GNC', 'Tipo de vendedor_concesionaria', 'Tipo de vendedor_particular', 'Tipo de vendedor_tienda', 
    # 'High_Mileage', 'Is_Old', 'Engine_Size_per_Year', 'Km_per_Cylinder', 'Motor_Age_Interaction', 'Mileage_Turbo_Interaction', 
    # 'Motor_Cylinder_Interaction', 'Año^2', 'Año Kilómetros', 'Año Edad', 'Año Km promedio por año', 'Año Cilindros', 
    # 'Año Motor', 'Kilómetros^2', 'Kilómetros Edad', 'Kilómetros Km promedio por año', 'Kilómetros Cilindros', 
    # 'Kilómetros Motor', 'Edad^2', 'Edad Km promedio por año', 'Edad Cilindros', 'Edad Motor', 'Km promedio por año^2', 
    # 'Km promedio por año Cilindros', 'Km promedio por año Motor', 'Cilindros^2', 'Cilindros Motor', 'Motor^2']]

    # mi input data tiene las siguientes columnas:
    # Marca, Modelo, Motor, Kilometraje, Año, Color, Tipo de combustible, Transmisión
    # Peugeot, 2008, 1.6, 23000, 2022, Blanco, Nafta, Automática

    # Agregar las columnas faltantes con valores por defecto
    input_data['Edad'] = 2024 - input_data['Año']
    input_data['Km promedio por año'] = input_data['Kilómetros'] / input_data['Edad']
    input_data['Cilindros'] = 4  # Suposición común, ajusta según tu necesidad
    input_data['Turbo'] = False

    # Establecer el tipo de vendedor como particular
    input_data['Tipo de vendedor_concesionaria'] = False
    input_data['Tipo de vendedor_particular'] = True
    input_data['Tipo de vendedor_tienda'] = False

    
    # Verificar la transmisión ingresada y establecer el valor correspondiente en True
    transmisiones = ['Automática', 'Automática secuencial', 'Manual', 'Semiautomática']

    for transmision in transmisiones:
        input_data[f'Transmisión_{transmision}'] = (input_data['Transmisión'] == transmision)

    # Establecer todas las marcas como False excepto la seleccionada
    marcas = [
        'Abarth', 'Alfa Romeo', 'Audi', 'BAIC', 'BMW', 'Chery', 'Chevrolet', 'Citroën', 'DS', 'Daihatsu', 'Dodge',
        'Fiat', 'Ford', 'Geely', 'Haval', 'Honda', 'Hyundai', 'Isuzu', 'JAC', 'Jaguar', 'Jeep', 'Jetour', 'Kia',
        'Land Rover', 'Lexus', 'Lifan', 'MINI', 'Mercedes-Benz', 'Mitsubishi', 'Nissan', 'Peugeot', 'Porsche',
        'Renault', 'Sandero', 'Ssangyong', 'Subaru', 'Suzuki', 'Toyota', 'Volkswagen', 'Volvo'
    ]
    for marca in marcas:
        input_data[f'Marca_{marca}'] = (input_data['Marca'] == marca)

    # Establecer todos los tipos de combustible como False excepto el seleccionado
    combustibles = [
        'Diésel', 'Eléctrico', 'GNC', 'Híbrido', 'Híbrido/Diesel', 'Híbrido/Nafta', 'Nafta', 'Nafta/GNC'
    ]
    for combustible in combustibles:
        input_data[f'Tipo de combustible_{combustible}'] = (input_data['Tipo de combustible'] == combustible)

    # Características adicionales
    input_data['Motor'] = input_data['Motor'].astype(float)
    input_data['Modelo'] = get_model_value(input_data['Modelo'].values[0])
    input_data['Color'] = input_data['Color'].astype('category').cat.codes

    input_data['High_Mileage'] = input_data['Kilómetros'] > 150000
    input_data['Is_Old'] = input_data['Edad'] > 10
    input_data['Engine_Size_per_Year'] = input_data['Motor'] / input_data['Edad']
    input_data['Km_per_Cylinder'] = input_data['Kilómetros'] / input_data['Cilindros']
    input_data['Motor_Age_Interaction'] = input_data['Motor'] * input_data['Edad']
    input_data['Mileage_Turbo_Interaction'] = input_data['Kilómetros'] * input_data['Turbo']
    input_data['Motor_Cylinder_Interaction'] = input_data['Motor'] * input_data['Cilindros']
    input_data['Año^2'] = input_data['Año'] ** 2
    input_data['Año Kilómetros'] = input_data['Año'] * input_data['Kilómetros']
    input_data['Año Edad'] = input_data['Año'] * input_data['Edad']
    input_data['Año Km promedio por año'] = input_data['Año'] * input_data['Km promedio por año']
    input_data['Año Cilindros'] = input_data['Año'] * input_data['Cilindros']
    input_data['Año Motor'] = input_data['Año'] * input_data['Motor']
    input_data['Kilómetros^2'] = input_data['Kilómetros'] ** 2
    input_data['Kilómetros Edad'] = input_data['Kilómetros'] * input_data['Edad']
    input_data['Kilómetros Km promedio por año'] = input_data['Kilómetros'] * input_data['Km promedio por año']
    input_data['Kilómetros Cilindros'] = input_data['Kilómetros'] * input_data['Cilindros']
    input_data['Kilómetros Motor'] = input_data['Kilómetros'] * input_data['Motor']
    input_data['Edad^2'] = input_data['Edad'] ** 2
    input_data['Edad Km promedio por año'] = input_data['Edad'] * input_data['Km promedio por año']
    input_data['Edad Cilindros'] = input_data['Edad'] * input_data['Cilindros']
    input_data['Edad Motor'] = input_data['Edad'] * input_data['Motor']
    input_data['Km promedio por año^2'] = input_data['Km promedio por año'] ** 2
    input_data['Km promedio por año Cilindros'] = input_data['Km promedio por año'] * input_data['Cilindros']
    input_data['Km promedio por año Motor'] = input_data['Km promedio por año'] * input_data['Motor']
    input_data['Cilindros^2'] = input_data['Cilindros'] ** 2
    input_data['Cilindros Motor'] = input_data['Cilindros'] * input_data['Motor']
    input_data['Motor^2'] = input_data['Motor'] ** 2

    # Ordenar las columnas según el encabezado
    columnas_ordenadas = ['Modelo', 'Año', 'Color', 'Motor', 'Kilómetros', 'Edad', 'Km promedio por año', 'Cilindros', 'Turbo', 'Transmisión_Automática', 'Transmisión_Automática secuencial', 'Transmisión_Manual', 'Transmisión_Semiautomática', 'Marca_Abarth', 'Marca_Alfa Romeo', 'Marca_Audi', 'Marca_BAIC', 'Marca_BMW', 'Marca_Chery', 'Marca_Chevrolet', 'Marca_Citroën', 'Marca_DS', 'Marca_Daihatsu', 'Marca_Dodge', 'Marca_Fiat', 'Marca_Ford', 'Marca_Geely', 'Marca_Haval', 'Marca_Honda', 'Marca_Hyundai', 'Marca_Isuzu', 'Marca_JAC', 'Marca_Jaguar', 'Marca_Jeep', 'Marca_Jetour', 'Marca_Kia', 'Marca_Land Rover', 'Marca_Lexus', 'Marca_Lifan', 'Marca_MINI', 'Marca_Mercedes-Benz', 'Marca_Mitsubishi', 'Marca_Nissan', 'Marca_Peugeot', 'Marca_Porsche', 'Marca_Renault', 'Marca_Sandero', 'Marca_Ssangyong', 'Marca_Subaru', 'Marca_Suzuki', 'Marca_Toyota', 'Marca_Volkswagen', 'Marca_Volvo', 'Tipo de combustible_Diésel', 'Tipo de combustible_Eléctrico', 'Tipo de combustible_GNC', 'Tipo de combustible_Híbrido', 'Tipo de combustible_Híbrido/Diesel', 'Tipo de combustible_Híbrido/Nafta', 'Tipo de combustible_Nafta', 'Tipo de combustible_Nafta/GNC', 'Tipo de vendedor_concesionaria', 'Tipo de vendedor_particular', 'Tipo de vendedor_tienda', 'High_Mileage', 'Is_Old', 'Engine_Size_per_Year', 'Km_per_Cylinder', 'Motor_Age_Interaction', 'Mileage_Turbo_Interaction', 'Motor_Cylinder_Interaction', 'Año^2', 'Año Kilómetros', 'Año Edad', 'Año Km promedio por año', 'Año Cilindros', 'Año Motor', 'Kilómetros^2', 'Kilómetros Edad', 'Kilómetros Km promedio por año', 'Kilómetros Cilindros', 'Kilómetros Motor', 'Edad^2', 'Edad Km promedio por año', 'Edad Cilindros', 'Edad Motor', 'Km promedio por año^2', 'Km promedio por año Cilindros', 'Km promedio por año Motor', 'Cilindros^2', 'Cilindros Motor', 'Motor^2']
    
    input_data = input_data[columnas_ordenadas]

    input_data.to_csv('input.csv', index=False)

    return input_data



