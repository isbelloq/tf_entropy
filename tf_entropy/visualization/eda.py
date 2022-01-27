import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
nltk.download("stopwords")
from nltk.corpus import stopwords
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number

def wordcloud_graph(text: str,title: str , stopword_language:str = 'spanish') -> WordCloud:
    '''
    Generación de nube de palabras
    
    Parameters
    ----------
    text: str
        Texto para generar la nube de palabras
    title: str
        Titulo de la gráfica
    stopword_language:str
        Idioma del texto para remoción de stopwords extraidos de nltk, por defecto 'spanish'
    
    Returns
    -------
    wordcloud: WordCloud
        Nube de palabras
    '''
    wordcloud = WordCloud(
            stopwords=stopwords.words(stopword_language),
            width=1600,
            height=800).generate(text)

    plt.figure(figsize=(25, 50))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, size = 50)
    plt.axis("off")
    plt.show()
    
    return(wordcloud)

def plot_top_n_ir(df:SparkDataFrame,document_col:str, ir_col:str, token_col:str = 'token',n:int =15, order:str = 'descending'):
    '''
    Función para graficar los n tokens más relevantes según un modelo de information retrieval
    
    Parameters
    ----------
    df:SparkDataFrame
        Conjunto de datos con tokens y resultados del modelo IR
    document_col:str
        Nombre de la columna de documentos
    ir_col:str
        Nombre de la columna con en resultado del IR
    token_col:str
        Nombre de la columna de tokens, por defecto `token`
    n:int =15
        Número de tokens más relevantes a representar, por defecto 15
    '''
    
    #Definición de ventana de tokens
    if order == 'descending':
        windowToken = Window.partitionBy(document_col).orderBy(col(ir_col).desc())
    elif order == 'ascending':
        windowToken = Window.partitionBy(document_col).orderBy(col(ir_col).asc())
    
    #Creación del dataframe a graficar
    top_n_df = df.\
        withColumn('row', row_number().over(windowToken)).\
        filter(col('row') <= n).\
        select(document_col, token_col, ir_col).\
        toPandas()

    fig = px.bar(top_n_df,
                x=ir_col,
                y=token_col,
                color=document_col,
                facet_col=document_col,
                facet_col_spacing=0.15,
                facet_row_spacing=0.04,
                facet_col_wrap=2,
                height=900, width=800,
                orientation='h')

    fig.update_xaxes(autorange=True, matches=None, showticklabels=True)
    
    if order == 'descending':
        fig.update_yaxes(autorange=True, matches=None, showticklabels=True, categoryorder = f'total ascending')
    elif order == 'ascending':
        fig.update_yaxes(autorange=True, matches=None, showticklabels=True, categoryorder = f'total descending')

    return fig 

    