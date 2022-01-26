from pydantic import BaseModel
from typing import List, Optional, Dict

class SparkDataFrameDic(BaseModel):
    """
    Clase de acceso al lago de datos.

    Attributes
    ----------
    int_columns : Optional[List[str]]
        Lista con los columnas de tipo entero
    float_columns : Optional[List[str]]
        Lista con los columnas de tipo flotante
    timestamp_columns : Optional[Dict[str, List[str]]]
        Diccionario de datos donde la llave es el formato de la fecha y el valor es
        una lista con el nombre de columnas
    categorical_columns : Optional[List[str]]
        Lista con las columans categóricas
    """
    #TODO: Encontrar la forma de pasar las columnas del conjunto de datos original y retornar
    #       la lista preprocesada: ejemplo enta la columna `Una Columna` y se retorna `una_columna`
    #FIXME
    #TODOC
    int_columns : Optional[List[str]]
    float_columns : Optional[List[str]]
    timestamp_columns : Optional[Dict[str, List[str]]]
    categorical_columns : Optional[List[str]]
    text_columns : Optional[List[str]]