import spacy
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import udf, col, lower, regexp_replace, trim, countDistinct, log, log2, lit, sum, explode, concat_ws, collect_list
from pyspark.sql.types import StructType, LongType,StringType,ArrayType, StructField
from typing import List, Tuple
from collections import Counter

def normalize_text(column:str) -> col:
    '''
    Función de normalización de texto para español
    
    Parameters
    ----------
    column:str
        Nombre de la columna a normalizar
    
    Returns
    -------
    normalize:col
        Columna normalizada
    '''
    
    #TODO: Por defecto solo se dejan caracteres alfabeticos y espacios.
    #      Hay que agregar opciones para incluir signos de puntuación y números
    #FIXME:
    #TODOC:
    
    normalize = col(column)
    #Paso del texto a minúsculos
    normalize = lower(normalize)
    #Normalización de diacríticos
    normalize = regexp_replace(normalize, r'à', 'á')
    normalize = regexp_replace(normalize, r'è', 'é')
    normalize = regexp_replace(normalize, r'ì', 'í')
    normalize = regexp_replace(normalize, r'ò', 'ó')
    normalize = regexp_replace(normalize, r'ù', 'ú')
        
    #Limpieza de caracteres
    normalize = regexp_replace(normalize, r'[^a-záéíóúüñ ]', ' ')
    
    #Normalización de espacios
    normalize = regexp_replace(normalize, r'\s+', ' ')
    normalize = trim(normalize)

    return normalize

@udf(returnType= ArrayType(
    StructType(
        [StructField("token",StringType()),
        StructField("n",LongType())])))
def spark_token_counter(text: str, spacy_model: str = 'es_core_news_sm') -> List[Tuple[str,int]]:
    '''
    Función de conteo de tokens en un texto texto
    
    Parameters
    ----------
    text : str
        Texto a tokenizar
    spacy_model : str
        Modelo de spaCy a uar, por defecto 'es_core_news_sm'
    
    Returns
    -------
    tokens: List[Tuple[str,int]]
        Lista con una tupla de (token,n)
    '''
    nlp = spacy.load(spacy_model)
    doc = nlp(text)
    tokens = [token.text for token in doc]
    tokens = [*Counter(tokens).items()]
    
    return tokens

def token_split(text: str, spacy_model: str = 'es_core_news_sm') -> List[str]:
    '''
    Función de tokenización de texto
    
    Parameters
    ----------
    text : str
        Texto a tokenizar
    spacy_model : str
        Modelo de spaCy a uar, por defecto 'es_core_news_sm'
    
    Returns
    -------
    tokens: List[str]
        Arregro con tokens
    '''
    nlp = spacy.load(spacy_model)
    doc = nlp(text)
    tokens = [token.text for token in doc]
        
    return tokens

def token_frequency(df:SparkDataFrame, document_col:str, token_col:str) -> SparkDataFrame:
    '''
    Función de conteo de frecuencia de tokens y de palabras en un documento
    
    Parameters
    ----------
    df:SparkDataFrame
        Conjunto de datos con tokens por documento
    document_col:str
        Nombre de la columna con los documentos
    token_col:str
        Nombre de la columna con los tokens
        
    Returns
    -------
    token_frequency:SparkDataFrame
        Dataset con las siguientes columnas adicionales: `n` frecuencia de aparición de la palabra por documento,
        `global_n` frecuencia de aparición de la palabra en el conjunto de datos, `total` número total
        de palabras por documento , `p_ij` la probabilidad condicional que se calcula como `p_ij = n/global_n` y
        `tf` con la frecuencia relativa de aparición del terímno en el documento, i.e. `tf = n/total`
    '''

    #Calculo de conteo de token global
    tokens_global = (
        df.\
        groupBy(token_col).\
        agg(sum('n').alias('global_n'))
    )
    
    #Calculo de número de palabras por documento
    word_count = (
        df.\
        groupBy(document_col).\
        agg(sum('n').alias('total'))
    )
    
    #Unión de dataframes
    token_frequency = (
        df.\
        join(tokens_global, on = [token_col], how = 'inner').\
        join(word_count, on = [document_col], how = 'inner').\
        withColumn('tf', col('n')/col('total')).\
        withColumn('p_ij', col('n')/col('global_n')).\
        orderBy(col('n').desc()))
    
    return token_frequency


def spark_tf_idf(df:SparkDataFrame, document_col:str, token_col:str) -> SparkDataFrame:
    '''
    Función de calculo de tf-idf
    
    Parameters
    ----------
    df:SparkDataFrame
        Conjunto de datos con token_frequency
    document_col:str
        Nombre de la columna con los documentos
    token_col:str
        Nombre de la columna con los tokens
        
    Returns
    -------
    tf_idf:SparkDataFrame
        Dataset con dos columnas adicionales idf y tf-idf por token
    '''
    
    #Calculo de n_t
    n_t = (
        df.\
        groupBy(token_col).\
        count().\
        select(token_col, col('count').alias('nt'))
    )

    #Calculo de n_d
    n_d = (
        df.\
        select(countDistinct(document_col).alias('nd')).\
        collect()[0]['nd'])

    #Calculo de tf-idf
    if_idf = (
        df.\
        join(n_t, on =[token_col], how='inner').\
        withColumn('idf', log(lit(n_d)/col('nt'))).\
        withColumn('tf-idf', col('tf') * col('idf')).\
        drop('nt').\
        orderBy(col('tf-idf').desc()))


    return if_idf

def spark_tf_ent(df:SparkDataFrame, document_col:str, token_col:str) -> SparkDataFrame:
    '''
    Función de calculo de tf-Entropy
    
    Parameters
    ----------
    df:SparkDataFrame
        Conjunto de datos con token_frequency
    document_col:str
        Nombre de la columna con los documentos
    token_col:str
        Nombre de la columna con los tokens
        
    Returns
    -------
    tf_Entropy:SparkDataFrame
        Dataset con la columna adicional de tf-Entropy por token
    '''
    
    #Calculo de número de documentos
    n_d = (
        df.\
        select(countDistinct(document_col).alias('nd')).\
        collect()[0]['nd'])

    #Calculo de Entropy
    entropy = (
        df.\
        withColumn('H_ij', - col('p_ij') * log2(col('p_ij'))).\
        groupBy(token_col).\
        agg(sum('H_ij').alias('entropy')).\
        withColumn('entropy', lit(1) - (col('entropy')/log2(lit(n_d)))))

    
    #Calculo de tf_Entropy
    tf_entropy = df.\
            join(entropy, on=[token_col], how='inner').\
            withColumn('tf-entropy', col('tf') * col('entropy')).\
            orderBy(col('tf-entropy').desc())


    return tf_entropy

def spark_log_ent(df:SparkDataFrame, document_col:str, token_col:str) -> SparkDataFrame:
    '''
    Función de calculo de log-Entropy
    
    Parameters
    ----------
    df:SparkDataFrame
        Conjunto de datos con token_frequency
    document_col:str
        Nombre de la columna con los documentos
    token_col:str
        Nombre de la columna con los tokens
        
    Returns
    -------
    tf_Entropy:SparkDataFrame
        Dataset con la columna adicional de log-Entropy por token
    '''
    
    #Calculo de número de documentos
    n_d = (
        df.\
        select(countDistinct(document_col).alias('nd')).\
        collect()[0]['nd'])

    #Calculo de Entropy
    entropy = (
        df.\
        withColumn('H_ij', - col('p_ij') * log2(col('p_ij'))).\
        groupBy(token_col).\
        agg(sum('H_ij').alias('entropy')).\
        withColumn('entropy', lit(1) - (col('entropy')/log2(lit(n_d)))))

    
    #Calculo de log_Entropy
    log_Entropy = df.\
            join(entropy, on=[token_col], how='inner').\
            withColumn('log-entropy', log2(col('tf')+lit(1)) * col('entropy')).\
            orderBy(col('log-entropy').desc())


    return log_Entropy

def count_token_by_doc(df:SparkDataFrame, document_col:str, text_col:str,spacy_model:str = 'es_core_news_sm' ) -> SparkDataFrame:
    '''
    Función de conteo de tokens por documento
    
    Parameters
    ----------
    df:SparkDataFrame
        Conjunto de datos que debe tener la columna de documentos y la columna de texto
    document_col:str
        Nombre de la columna de documentos
    text_col:str
        Nombre de la columna del texto
    spacy_model : str
        Modelo de spaCy a uar, por defecto 'es_core_news_sm'
    
    Returns
    -------
    token_count: SparkDataFrame
        Conjunto de datos con tres columnas, `document_col` nombre de la columna del documento,
        `token` token en el documento y `count` frecuencia de aparición del token en el documento
    '''
    token_count = df.\
    withColumn(text_col, normalize_text(text_col)).\
    groupby(document_col).\
    agg(concat_ws(' ',collect_list(text_col)).alias(text_col)).\
    withColumn('token_count', explode(spark_token_counter(text_col, lit(spacy_model)))).\
    select(document_col,
           col('token_count').getItem('token').alias('token'),
           col('token_count').getItem('n').alias('n'))


    return token_count