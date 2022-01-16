from pandas import DataFrame
import pandas as pd
import os

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession



def load_data_to_dataframe(lake_path:str, file_name: str, **kwArgs) -> DataFrame:
    """
    Funcion de carga de informacion de datos a dataframe

    Parameters
    ----------
    lake_path:str
        Ruta de almacenamiento del arachivo
    file_name: str
        Nombre del archivo de lectura
    **kwArgs
        Parametros adicionales que dependen de pandas

    Returns
    -------
    df : DataFrame
        DataFrame con la informacion cargada
    """
    # TODO:
    # FIXME:
    # TODOC:

    #Definicion de ruta y extension del archivo
    file_path = os.path.join(lake_path, file_name)
    file_extention = os.path.splitext(file_name)[1][1:]
    
    #Constucción de expresión a evaluar
    eval_str = f"pd.read_{file_extention}('{file_path}'"
    for key, value in kwArgs.items():
        eval_str += f", {key} = '{value}'"
    eval_str += ')'

    #Carga de informacion
    df = eval(eval_str)
    
    return(df)

def save_dataframe(dataframe:DataFrame, lake_path:str, file_name: str):
    """
    Funcion de almacenamiento de infromación

    Parameters
    ----------
    dataframe:DataFrame
        Conjunto de datos a almacenar
    lake_path:str
        Ruta de almacenamiento del arachivo
    file_name: str
        Nombre del archivo de lectura, es necesario que tenga la extención de almacenamiento ejemplo `datos.csv` o `datos.parquet`
    """
    # TODO:
    # FIXME:
    # TODOC:

    #Definicion de ruta y extension del archivo
    file_path = os.path.join(lake_path, file_name)
    file_extention = os.path.splitext(file_name)[1][1:]

    #Carga de informacion
    exec(f"df.to_{file_extention}('{file_path}', index = False)", {'df':dataframe})


def load_data_to_spark(lake_path:str, file_name: str, **kwArgs) -> SparkDataFrame:
    """
    Funcion de carga de informacion de datos a un spark dataframe

    Parameters
    ----------
    lake_path:str
        Ruta de almacenamiento del arachivo
    file_name: str
        Nombre del archivo de lectura
    **kwArgs
        Valoser de 'option' permitidos por pyspark

    Returns
    -------
    df : SparkDataFrame
        DataFrame con la informacion cargada en spark
    """
    # TODO:
    # FIXME:
    # TODOC:

    #Definicion de ruta y extension del archivo
    file_path = os.path.join(lake_path, file_name)
    file_extention = os.path.splitext(file_name)[1][1:]
    
    spark = SparkSession.builder.getOrCreate()
    
    #Constucción de expresión a evaluar
    eval_str = f"spark.read.format('{file_extention}')"
    for key, value in kwArgs.items():
        eval_str += f".option('{key}', '{value}')"
    eval_str += f".load('{file_path}')"

    #Carga de informacion
    df = eval(eval_str, {'spark':spark})
    
    return(df)

def save_spark(dataframe:SparkDataFrame, lake_path:str, file_name: str, mode:str = 'overwrite', **kwArgs):
    """
    Funcion de almacenamiento de infromación usando spark

    Parameters
    ----------
    dataframe:SparkDataFrame
        Conjunto de datos a almacenar en spark
    lake_path:str
        Ruta de almacenamiento del arachivo
    file_name: str
        Nombre del archivo de lectura, es necesario que tenga la extención de almacenamiento ejemplo `datos.csv` o `datos.parquet`
    mode : str
        Modo de almacenamiento de información, por defecto 'overwrite'
    **kwArgs
        Valoser de 'option' permitidos por pyspark
    """
    # TODO:
    # FIXME:
    # TODOC:

    #Definicion de ruta y extension del archivo
    file_path = os.path.join(lake_path, file_name)
    file_extention = os.path.splitext(file_name)[1][1:]

    #Constucción de expresión a evaluar
    exec_str = f"df.write"
    for key, value in kwArgs.items():
        exec_str += f".option('{key}', '{value}')"
    exec_str += f".format('{file_extention}').mode('{mode}').save('{file_path}')"


    #Ejecución de scrip de almacenamiento
    exec(exec_str, {'df':dataframe})