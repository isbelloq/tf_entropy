from pandas import DataFrame
import pandas as pd
import os

def load_data_to_dataframe(lake_path:str, file_name: str) -> DataFrame:
    """
    Funcion de carga de informacion de datos a dataframe

    Parameters
    ----------
    lake_path:str
        Ruta de almacenamiento del arachivo
    file_name: str
        Nombre del archivo de lectura

    Returns
    -------
    df : DataFrame
        DataFrame con la informacion cargada
    """
    # TODO: Cambio separador por defecto de ',' a algo paramétrico
    # TODO: Cambio para recivir **kwArgs
    # FIXME:
    # TODOC:

    #Definicion de ruta y extension del archivo
    file_path = os.path.join(lake_path, file_name)
    file_extention = os.path.splitext(file_name)[1][1:]

    #Carga de informacion
    df = eval(f"pd.read_{file_extention}('{file_path}', sep = ',', encoding='windows-1252')")
    
    return(df)