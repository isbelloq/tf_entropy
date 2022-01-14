# -*- coding: utf-8 -*-
from tf_entropy.environment.base import get_data_paths
from tf_entropy.database.io import load_data_to_dataframe
import os

# # Carga de datos

# +
#Carga de paths del lago de datos
datalake = get_data_paths()

#Definición de path de datos
raw_file_path = os.path.join(datalake.raw_path, 'toymodel/data/DATE=2022-01-12')
raw_file_name = 'austen_books.csv'

#Carga de datos raw
df = load_data_to_dataframe(lake_path=raw_file_path, file_name=raw_file_name)
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

# # Homologación de texto

df

df.loc[df['book'] == 'Emma',:].tail(20)

df.dropna().loc[df['book'] == 'Pride & Prejudice',['book','chapter']].drop_duplicates()
