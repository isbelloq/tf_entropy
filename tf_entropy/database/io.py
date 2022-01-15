from pandas import DataFrame
import pandas as pd
import os

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

def save_dataframe(dataframe:DataFrame, lake_path:str, file_name: str, format: str = 'parquet'):
    """
    Funcion de almacenamiento de infromación

    Parameters
    ----------
    dataframe:DataFrame
        Conjunto de datos a almacenar
    lake_path:str
        Ruta de almacenamiento del arachivo
    file_name: str
        Nombre del archivo de lectura
    format: str
        Formato de almacenamiento, por defecto 'parquet'
    """
    # TODO:
    # FIXME:
    # TODOC:

    #Definicion de ruta y extension del archivo
    file_path = os.path.join(lake_path, file_name)

    #Carga de informacion
    exec(f"df.to_{format}('{file_path}', index = False)", {'df':dataframe})