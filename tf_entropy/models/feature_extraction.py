from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import udf, struct, collect_list, map_from_entries
from pyspark.sql.types import DoubleType
from pandas import DataFrame
from typing import Dict,Tuple

@udf(returnType=DoubleType())
def compute_score(text:str,data_model:Dict[str,float]) -> float:
    '''
    Función de computo de score
    
    Parameters
    ----------
    text:str
        Texto
    data_model:Dict[str,float]
        Diccionario donde la llave es el token el valor es el peso del token en el modelo


    Returns
    -------
    score:float
        Valor total del score
    '''
    score = 0.
    for token in text.split():
        if token in data_model.keys():
            score += data_model[token]
        
    return(score)

def create_map_model(df: SparkDataFrame, document_col:str, token_col:str, model_col:str) ->SparkDataFrame:
    '''
    Función de map para el modelo
    
    Parameters
    ----------
    df: SparkDataFrame
        Conjunto de datos con el modelo
    document_col:str
        Nombre de la columna que identifica el documento
    token_col:str
        Nombre de la columna con el token
    model_col:str
        Nombre de la columna con los pesos del modelo

    Returns
    -------
    model:SparkDataFrame
        Conjunto de datos con dos columnas, la columna del documento y la columna
        del mapeo
    '''
    model = (df.\
            select(document_col, struct(token_col, model_col).alias('map')).\
            groupby(document_col).\
            agg(collect_list('map').alias('map')).\
            select(document_col, map_from_entries('map').alias('model_map')))
    return(model)

def dataframe_score(df:SparkDataFrame, df_map:SparkDataFrame, id_col:str,document_col:str, text_col:str)->SparkDataFrame:
    '''
    Función de map para el modelo
    
    Parameters
    ----------
    df:SparkDataFrame
    df_map:SparkDataFrame
    id_col:str
    document_col:str
    text_col:str

    Returns
    -------
    model:SparkDataFrame
        Conjunto de datos con dos columnas, la columna del documento y la columna
        del mapeo
    '''
    #TODO
    #FIXME
    #TODOC: Falta la documentación :( estoy cansado y hay que seguir
    model = df.join(df_map, on =[document_col], how='inner')
    model = model.select(id_col, compute_score(text_col,'model_map').alias('score'))
    return(model)

def features_lables_split(df:DataFrame, y_label:str,id_label:str ='uid') -> Tuple[DataFrame,DataFrame]:
    '''
    Función separación del datafrme de features y lables
    
    Parameters
    ----------
    df:DataFrame
    y_label:str
    id_label:str ='uid'
    
    Returns
    -------
    df_X,df_y:Tuple[DataFrame,DataFrame]
    '''
    df_X = df.drop([id_label, y_label], axis=1) 
    df_y = df.loc[:,y_label]
    return(df_X,df_y)