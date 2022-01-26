# -*- coding: utf-8 -*-
# +
from tf_entropy.environment.base import get_data_paths
from tf_entropy.environment.spark import create_spark_session
from tf_entropy.database.io import load_data_to_spark, save_spark
from tf_entropy.preprocessing.pipeline import normalization_pipeline
from tf_entropy.datamodels.datadic import SparkDataFrameDic
import os

spark = create_spark_session()
# -

# # Carga de datos

# +
#Carga de paths del lago de datos
datalake = get_data_paths()

#Definición de path de datos
raw_file_path = os.path.join(datalake.raw_path, 'datos_abiertos/secop_i/DATE=2022-01-25')
raw_file_name = 'secop_i.csv'

bronze_file_path = os.path.join(datalake.bronze_path, 'datos_abiertos/secop')
bronze_file_name = 'secop_i.delta'

#Carga de datos raw
secop = load_data_to_spark(
        lake_path=raw_file_path,
        file_name=raw_file_name,
        sep = ',',
        header='true'
)
# -

#Definición de diccionario
secop_i_dic = SparkDataFrameDic(
    int_columns = ['anno_firma_del_contrato', 'plazo_de_ejec_del_contrato',
                   'tiempo_adiciones_en_dias','tiempo_adiciones_en_meses'],
    float_columns = ['cuantia_contrato', 'valor_total_de_adiciones',
                    'valor_contrato_con_adiciones'],
    timestamp_columns = {'yyyy-MM-dd':['fecha_ini_ejec_contrato'],
                        'yyyy-MM-dd HH:mm:ss': ['fecha_fin_ejec_contrato']},
    categorical_columns = ['nivel_entidad', 'regimen_de_contratacion', 'id_clase',
                            'rango_de_ejec_del_contrato', 'origen_de_los_recursos',
                            'moneda', 'municipio_entidad', 'departamento_entidad'],
    text_columns = ['objeto_del_contrato_a_la_firma']
)
#Ejecución de pipeline de preprocesamiento
secop = normalization_pipeline(df=secop,
                               datadic = secop_i_dic)

# # Almacenamiento en bronce

# +
#Creación de path si no existe
if not os.path.exists(bronze_file_path):
    os.makedirs(bronze_file_path)

#Almacenamiento en delta
save_spark(dataframe=secop, lake_path=bronze_file_path, file_name=bronze_file_name)
