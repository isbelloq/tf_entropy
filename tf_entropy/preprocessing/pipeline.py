from tf_entropy.preprocessing.text import token_frequency, spark_tf_idf, spark_tf_ent, spark_log_ent
from pyspark.sql import DataFrame as SparkDataFrame
from tf_entropy.datamodels.datadic import SparkDataFrameDic
from tf_entropy.preprocessing.dataset import columns_df_norm, cast_int, cast_float, cast_timestamp, normalize_cat
from pyspark.sql.functions import explode, lit

def tfidf_pipeline(df: SparkDataFrame, token_col:str, document_col:str) -> SparkDataFrame:
    '''
    Pipeline para obtención de tf-idf
    
    Parameters
    ----------
    df: SparkDataFrame
        Dataframe de procesamiento, es importante una columna de tokens y una columna de documentos
    token_col:str
        Nombre de la columna con los tokens
    document_col:str
        Nombre de la columna con los documentos
        
    Returns
    -------
    tfidf_df:SparkDataFrame
        Dataset de tf-idf
    '''
    
    #Definición de flujo de procesamiento 
    tfidf_df = token_frequency(df=df, document_col=document_col, token_col=token_col)
    tfidf_df = spark_tf_idf(df=tfidf_df, document_col=document_col, token_col=token_col)

    return tfidf_df

def tfent_pipeline(df: SparkDataFrame, token_col:str, document_col:str) -> SparkDataFrame:
    '''
    Pipeline para obtención de tf-Entropy
    
    Parameters
    ----------
    df: SparkDataFrame
        Dataframe de procesamiento, es importante una columna de tokens y una columna de documentos
    token_col:str
        Nombre de la columna con los tokens
    document_col:str
        Nombre de la columna con los documentos
        
    Returns
    -------
    tfent_df:SparkDataFrame
        Dataset de tf-Entropy 
    '''
    
    #Definición de flujo de procesamiento 
    tfent_df = token_frequency(df=df, document_col=document_col, token_col=token_col)
    tfent_df = spark_tf_ent(df=tfent_df, document_col=document_col, token_col=token_col)

    return tfent_df

def logent_pipeline(df: SparkDataFrame, token_col:str, document_col:str) -> SparkDataFrame:
    '''
    Pipeline para obtención de log-Entropy
    
    Parameters
    ----------
    df: SparkDataFrame
        Dataframe de procesamiento, es importante una columna de tokens y una columna de documentos
    token_col:str
        Nombre de la columna con los tokens
    document_col:str
        Nombre de la columna con los documentos
        
    Returns
    -------
    logent:SparkDataFrame
        Dataset de tf-idf
    '''
    
    #Definición de flujo de procesamiento 
    logent_df = token_frequency(df=df, document_col=document_col, token_col=token_col)
    logent_df = spark_log_ent(df=logent_df, document_col=document_col, token_col=token_col)

    return logent_df

def normalization_pipeline(df: SparkDataFrame, datadic: SparkDataFrameDic) -> SparkDataFrame:
    '''
    Pipeline de normalización del conjunto de datos
    
    Parameters
    ----------
    df: SparkDataFrame
        Dataset a normalizar
    datadic: SparkDataFrameDic
        Diccionario de parámetro
    Returns
    -------
    df:SparkDataFrame
        Dataset con columnas casteadas y categorias normalizadas
    '''
    
    #Normalización de columnas
    df = columns_df_norm(df=df)

    #Casteo de informacion
    df = cast_int(df=df, datadic=datadic)
    df = cast_float(df=df, datadic=datadic)
    df = cast_timestamp(df=df, datadic=datadic)

    #Normalización de categorías
    df = normalize_cat(df=df, datadic=datadic)

    return df