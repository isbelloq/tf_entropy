from tf_entropy.preprocessing.text import normalize_text, spark_split, token_frequency, spark_tf_idf, spark_tf_ent, spark_log_ent
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import explode, lit

def token_pipeline(df: SparkDataFrame, text_col:str, spacy_model:str = 'es_core_news_sm') -> SparkDataFrame:
    '''
    Pipeline para obtención de tokens
    
    Parameters
    ----------
    df: SparkDataFrame
        Dataframe con el texto
    text_col:str
        Nombre de la columna con el texto
    spacy_model:str
        Modelo de spaCy para tokenizar, por defecto `es_core_news_sm`
        
    Returns
    -------
    df_tokens:SparkDataFrame
        Dataset con la columna de tokens
    '''
    
    #Definición de flujo de procesamiento 
    prepros_text = normalize_text(text_col)
    prepros_text = spark_split(prepros_text, lit(spacy_model))
    prepros_text = explode(prepros_text)

    df_tokens = df.withColumn('token', prepros_text).drop(text_col)

    return df_tokens

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