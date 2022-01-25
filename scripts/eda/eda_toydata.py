# -*- coding: utf-8 -*-
# +
from tf_entropy.environment.base import get_data_paths
from tf_entropy.environment.spark import create_spark_session
from tf_entropy.database.io import load_data_to_spark, save_spark
from tf_entropy.preprocessing.text import normalize_text
from tf_entropy.preprocessing.pipeline import token_pipeline, tfidf_pipeline, tfent_pipeline
from tf_entropy.visualization.eda import wordcloud_graph, plot_top_n_ir

from pyspark.sql.functions import concat_ws, collect_list
import plotly.express as px
import os

spark = create_spark_session()

# +
#Carga de paths del lago de datos
datalake = get_data_paths()

#Definición de path de almacenamiento de datos
bronze_path = os.path.join(datalake.bronze_path, 'toymodel/data/YEAR=2022/MONTH=01/DAY=14_22-03')
file_name = 'austen.delta'

silver_path = os.path.join(datalake.silver_path, 'toymodel/data/frecuencia_palabras')
silver_file_name = 'frecuencia_palabras.delta'

fig_path = os.path.join(os.environ['FIGURE_PATH'], 'eda')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

#Lectura de infromación
austen_df = load_data_to_spark(lake_path = bronze_path, file_name=file_name)
# -
# # Exploración de datos

# ## Capítulos por libro

austen_chap = austen_df.\
select('book','chapter').\
distinct().\
groupby('book').\
count().\
toPandas()

fig = px.bar(
    austen_chap,
    x='book',
    y='count',
    labels={'book':'Libros', 'count':'No. Capítulos'})
fig.show()

# ## Nube de palabas por libro

austen_wc = austen_df.\
select('book', normalize_text('text').alias('text')).\
groupby('book').\
agg(concat_ws(' ',collect_list('text')).alias('text')).\
toPandas()

for index, row in austen_wc.iterrows():
    plot_by_book = wordcloud_graph(
                            text = row['text'],
                            title= row['book'],
                            stopword_language = 'english')
    plot_by_book.to_file(os.path.join(fig_path, 'wordcloud_{}.png'.format(row['book'])))

# ## Frecuencia de aparición de palablas

austen_tokens = token_pipeline(df=austen_df,
                               text_col = 'text',
                               spacy_model = 'en_core_web_sm')

# +
# save_spark(dataframe=austen_tokens, lake_path=silver_path, file_name=silver_file_name)
# austen_tokens = load_data_to_spark(lake_path = silver_path, file_name=silver_file_name)

# +
#Pipeline tf-idf
austen_tfidf = tfidf_pipeline(df=austen_tokens,
                              document_col='book',
                              token_col='token')

#Pipeline tf-Entropy
austen_tfEnt = tfent_pipeline(df=austen_tokens,
                      document_col='book',
                      token_col='token')
# -

#Distribución de la frecuencia de los términos en las novelas de Jane Austen 
fig = px.histogram(austen_tfidf.toPandas(),
             x='tf',
             color="book",
             facet_col='book',
             facet_col_wrap=2)
fig.update_xaxes(range=[0, 0.0009])
fig.update_yaxes(autorange=True)
fig.show()

#Palabras con mayor peso de tf-idf en cada una de las novelas de Jane Austen 
plot_top_n_ir(df=austen_tfidf,
             document_col='book',
             ir_col='tf-idf')

#Palabras con mayor peso de tf-logEntropy en cada una de las novelas de Jane Austen 
plot_top_n_ir(df=austen_tfEnt,
             document_col='book',
             ir_col='tf-Entropy')
