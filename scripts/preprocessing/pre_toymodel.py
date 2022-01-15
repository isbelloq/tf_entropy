# -*- coding: utf-8 -*-
from tf_entropy.environment.base import get_data_paths
from tf_entropy.database.io import load_data_to_dataframe, save_dataframe
import os

# # Carga de datos

# +
#Carga de paths del lago de datos
datalake = get_data_paths()

#Definición de path de datos
raw_file_path = os.path.join(datalake.raw_path, 'toymodel/data/DATE=2022-01-12')
raw_file_name = 'austen_books.csv'

bronze_file_path = os.path.join(datalake.bronze_path, 'toymodel/data/YEAR=2022/MONTH=01/DAY=14_22-03')
bornze_file_name = 'austen.parquet'

#Carga de datos raw
df = load_data_to_dataframe(
        lake_path=raw_file_path,
        file_name=raw_file_name,
        sep = ',',
        encoding='windows-1252')
# -

# # Extracción de capítulos (solo válido para toymodel)

# +
#Extracción de volumne (Valido solo para book = Emma)
df['volume'] = df['text'].str.contains('^VOLUME', regex=True, na=False)
df['volume'] = df.loc[:,['text','volume']].apply(lambda x: x[0] if x[1] else None , axis = 1)
df['volume'] = df.groupby(['book'])['volume'].ffill()

#Extracción de capítulos
df['chapter'] = df['text'].str.contains('^CHAPTER|^Chapter', regex=True, na=False)
df['chapter'] = df.loc[:,['text','chapter']].apply(lambda x: x[0] if x[1] else None , axis = 1)
df['chapter'] = df.groupby(['book'])['chapter'].ffill()

#Concatenación volumen y capítulo
df['chapter'] = (df['volume']+'-').fillna('') + df['chapter'] 

#Borrado de columna auxiliar de volumne
df.drop(['Unnamed: 0','volume'], axis=1, inplace=True)

#Borrado de valores nulos y reset del index
df.dropna(inplace=True)
df.reset_index(drop = True, inplace = True)
# -

# # Almacenamiento en bronce

# +
#Creación de path si no existe
if bronze_file_path not in [*map(lambda x: x[0], os.walk(datalake.bronze_path))]:
    os.makedirs(bronze_file_path)

#Almacenamiento en parquet
save_dataframe(dataframe=df, lake_path=bronze_file_path, file_name=bornze_file_name)
