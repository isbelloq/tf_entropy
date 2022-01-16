# -*- coding: utf-8 -*-
# +
from tf_entropy.environment.base import get_data_paths
from tf_entropy.database.io import load_data_to_spark, save_spark
import os

import pyspark
from delta import *

#Configuración de Spark
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
builder = pyspark.sql.SparkSession.builder.appName("MyApp") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = configure_spark_with_delta_pip(builder).getOrCreate()
# -

# # Carga de datos

# +
#Carga de paths del lago de datos
datalake = get_data_paths()

#Definición de path de datos
raw_file_path = os.path.join(datalake.raw_path, 'toymodel/data/DATE=2022-01-12')
raw_file_name = 'austen_books.csv'

bronze_file_path = os.path.join(datalake.bronze_path, 'toymodel/data/YEAR=2022/MONTH=01/DAY=14_22-03')
bronze_file_name = 'austen.delta'

#Carga de datos raw
df = load_data_to_spark(
        lake_path=raw_file_path,
        file_name=raw_file_name,
        sep = ',',
        encoding='windows-1252',
        header='true'
)
df = df.toPandas()
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
df.drop(['_c0','volume'], axis=1, inplace=True)

#Borrado de valores nulos y reset del index
df.dropna(inplace=True)
df.reset_index(drop = True, inplace = True)
# -

# # Almacenamiento en bronce

# +
#Creación de path si no existe
if bronze_file_path not in [*map(lambda x: x[0], os.walk(datalake.bronze_path))]:
    os.makedirs(bronze_file_path)

#Almacenamiento en delta
df = spark.createDataFrame(df)
save_spark(dataframe=df, lake_path=bronze_file_path, file_name=bronze_file_name)
