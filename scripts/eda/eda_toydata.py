# -*- coding: utf-8 -*-
from tf_entropy.environment.base import get_data_paths
from tf_entropy.database.io import load_data_to_dataframe
import os

# +
#Carga de paths del lago de datos
datalake = get_data_paths()

#Definición de path de almacenamiento de datos
raw_path = os.path.join(datalake.raw_path, 'toymodel/data/DATE=2022-01-12')
file_name = 'austen_books.csv'

#Lectura de infromación
austen_df = load_data_to_dataframe(lake_path = raw_path, file_name=file_name)
# -


