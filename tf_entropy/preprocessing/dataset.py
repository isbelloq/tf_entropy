from pyspark.sql import DataFrame as SparkDataFrame
from tf_entropy.datamodels.datadic import SparkDataFrameDic
from pyspark.sql.functions import col, regexp_replace, to_timestamp, upper, trim
import re


def columns_df_norm(df:SparkDataFrame) -> SparkDataFrame:
    '''
    Función de normalización de normalización de nombres de columna

    Parameters
    ----------
    df: SparkDataFrame
        Dataframe con columnas a normalizar
        
    Returns
    -------
    df:SparkDataFrame
        Dataframe con columnas normalizadas
    '''

    columns_new = df.columns
    columns_new = [re.sub(r'[^a-z]', '_', column.lower()) for column in columns_new]
    columns_new = [re.sub(r'_+', '_', column) for column in columns_new]
    columns_new = [column.strip('_') for column in columns_new]

    for old, new in zip(df.columns, columns_new):
        df = df.withColumnRenamed(old, new)
    
    return(df)

def cast_int(df: SparkDataFrame, datadic: SparkDataFrameDic):
    '''
    Función de casteo a entero

    Parameters
    ----------
    df: SparkDataFrame
        Conjunto de datos a castear
    datadic: SparkDataFrameDic
        Diccionario de datos
        
    Returns
    -------
    df:SparkDataFrame
        Conjunto de datos con columnas en de tipo entero
    '''
    for int_col in datadic.int_columns:
        df = df.\
        withColumn(int_col, regexp_replace(col(int_col), r',', r'.')).\
        withColumn(int_col, regexp_replace(col(int_col), r'[^0-9.]', r'')).\
        withColumn(int_col, col(int_col).cast('int'))
        
    return(df)
    
def cast_float(df: SparkDataFrame, datadic: SparkDataFrameDic):
    '''
    Función de casteo a entero

    Parameters
    ----------
    df: SparkDataFrame
        Conjunto de datos a castear
    datadic: SparkDataFrameDic
        Diccionario de datos
        
    Returns
    -------
    df:SparkDataFrame
        Conjunto de datos con columnas en de tipo flotante
    '''
    for float_col in datadic.float_columns:
        df = df.\
        withColumn(float_col, regexp_replace(col(float_col), r',', r'.')).\
        withColumn(float_col, regexp_replace(col(float_col), r'[^0-9.]', r'')).\
        withColumn(float_col, col(float_col).cast('float'))
        
    return(df)

    
def cast_timestamp(df: SparkDataFrame, datadic: SparkDataFrameDic):
    '''
    Función de casteo a entero

    Parameters
    ----------
    df: SparkDataFrame
        Conjunto de datos a castear
    datadic: SparkDataFrameDic
        Diccionario de datos
        
    Returns
    -------
    df:SparkDataFrame
        Conjunto de datos con columnas en de tipo timestamp
    '''
    for ts_format, ts_cols in datadic.timestamp_columns.items():
        for ts_col in ts_cols:
            df = df.withColumn(ts_col, to_timestamp(col(ts_col), ts_format))
        
    return(df)

def normalize_cat(df: SparkDataFrame, datadic: SparkDataFrameDic):
    '''
    Función de normalización de categorias

    Parameters
    ----------
    df: SparkDataFrame
        Conjunto de datos a castear
    datadic: SparkDataFrameDic
        Diccionario de datos
        
    Returns
    -------
    df:SparkDataFrame
        Conjunto de datos con categorias normalizadas
    '''
    for category_cols in datadic.categorical_columns:
        df = df.\
            withColumn(category_cols, upper(col(category_cols))).\
            withColumn(category_cols, regexp_replace(col(category_cols), r'Á', r'A')).\
            withColumn(category_cols, regexp_replace(col(category_cols), r'É', r'E')).\
            withColumn(category_cols, regexp_replace(col(category_cols), r'Í', r'I')).\
            withColumn(category_cols, regexp_replace(col(category_cols), r'Ó', r'O')).\
            withColumn(category_cols, regexp_replace(col(category_cols), r'Ú', r'U')).\
            withColumn(category_cols, regexp_replace(col(category_cols), r'Ü', r'U')).\
            withColumn(category_cols, regexp_replace(col(category_cols), r'[^A-Z0-9Ñ ]', r' ')).\
            withColumn(category_cols, regexp_replace(col(category_cols), r' +', r' ')).\
            withColumn(category_cols, trim(col(category_cols)))
    return(df)
    