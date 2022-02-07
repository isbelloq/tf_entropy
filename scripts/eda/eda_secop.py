# -*- coding: utf-8 -*-
# +
from tf_entropy.environment.base import get_data_paths
from tf_entropy.environment.spark import create_spark_session
from tf_entropy.database.io import load_data_to_spark, save_spark
from tf_entropy.preprocessing.text import normalize_text, count_token_by_doc
from tf_entropy.preprocessing.dataset import text_normalize_cat
from tf_entropy.preprocessing.pipeline import tfidf_pipeline, tfent_pipeline
from tf_entropy.visualization.eda import wordcloud_graph, plot_top_n_ir


from pyspark.sql.window import Window
from pyspark.sql.functions import concat_ws, collect_list, expr, when, col, row_number, count, mean, sum
import matplotlib.pyplot as plt
import plotly.express as px
import geopandas as gpd
import os

spark = create_spark_session()

# +
#Carga de paths del lago de datos
datalake = get_data_paths()

#Definición de path de almacenamiento de datos
bronze_path = os.path.join(datalake.bronze_path, 'datos_abiertos/secop')
file_name = 'secop_i.delta'

silver_path = os.path.join(datalake.silver_path, 'datos_abiertos/secop/frecuencia_palabras')
silver_file_name = 'frecuencia_palabras_por_clase.delta'

geo_path = os.path.join(datalake.raw_path, 'geometry/colombia/DATE=2022-01-26')
geo_file_name = 'Limite Departamental.shp'
geo_file_path = os.path.join(geo_path, geo_file_name)


fig_path = os.path.join(os.environ['FIGURE_PATH'], 'eda/secop')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

#Lectura de infromación
secop = load_data_to_spark(lake_path = bronze_path, file_name=file_name)
secop = secop.withColumn('departamento_entidad',
                         when(col('departamento_entidad') == 'SAN ANDRES PROVIDENCIA Y SANTA CATALINA', 'SAN ANDRES').\
                        otherwise(col('departamento_entidad')))
# -
# # Preparación del conjunto de datos

# ## Identificación de clases

# +
#Diccionario de clases
id_clase_dic = {
    'bono_medio_de_pago' : ['801417']
    ,'call_center' : ['831115','841315', '841316']
    ,'disenho_arquitectonico' : ['801116', '811015']
    ,'fiducia_publica' : ['931515', '841217']
    ,'servicios_autorizados_sociedades_fiduciarias' : ['841218', '841217', '801318', '801317']
    ,'firma_digital' : ['321016', '432332']
    ,'inventoria_obras_publicas' : ['801016', '811015']
    ,'emergencias_medicas_domicilio' : ['851016']
    ,'senhalizacion_carreteras' : ['551217','721410']
    ,'servicios_notariales' : ['931515']
    ,'tiquetes_bus' : ['781118']
}

#Construcción de expresión de evaluación
id_clase_expr = 'CASE'

for clase, id_clase in id_clase_dic.items():
    id_clase_expr = id_clase_expr+" WHEN id_clase IN ('"+"','".join(id_clase)+"') THEN '" + clase + "'"

id_clase_expr += "ELSE 'no_clase' END"

secop = secop.withColumn('clase', expr(id_clase_expr))
# -

# ## Construcción de columna binaria de adiciones

secop = secop.withColumn('binary_adiciones', when(((col('tiempo_adiciones_en_dias')>0)
                                                  |(col('tiempo_adiciones_en_meses')>0)
                                                  |(col('valor_total_de_adiciones')>0)), 1
                                                 ).otherwise(0))

# ## Filtros de datos

secop = secop.\
filter(
    (col('clase') != 'disenho_arquitectonico')
    &(col('moneda') == 'PESO COLOMBIANO')
    &(col('cuantia_contrato') > 0)
    &(col('valor_contrato_con_adiciones') > 0)
)

print (secop.\
summary().\
toPandas().\
set_index('summary').\
T.\
to_latex())

# ## Almacenamiento para modelo

# +
secop_modelo = secop.select('uid',
                           'nivel_entidad',
                           'regimen_de_contratacion',
                           'valor_contrato_con_adiciones',
                           'objeto_del_contrato_a_la_firma',
                           'origen_de_los_recursos',
                           'departamento_entidad',
                           'clase',
                           'binary_adiciones')

save_spark(dataframe=secop_modelo,
           lake_path=silver_path,
           file_name='secop_entrenamiento.delta',
           overwriteSchema='true')
# -

# # Exploración de datos

# +
#Número de contratos por año de fima del contrato
secop_clase_conteo = secop.\
groupby('anno_firma_del_contrato', 'clase').\
count().\
toPandas()

fig = px.bar(
    secop_clase_conteo,
    x='anno_firma_del_contrato',
    y='count',
    color = 'clase',
    labels={
        'anno_firma_del_contrato' : 'Año firma del contrato',
        'count' : 'No. de contratos',
        'clase' : 'Tipo de producto'
    }
)
fig.show()

# +
#Monto de contratos por año de fima del contrato
secop_clase_suma = secop.\
groupby('anno_firma_del_contrato', 'clase').\
agg(sum('valor_contrato_con_adiciones').alias('total_contrato')).\
toPandas()

fig = px.bar(
    secop_clase_suma,
    x='anno_firma_del_contrato',
    y='total_contrato',
    color = 'clase',
    labels={
        'anno_firma_del_contrato' : 'Año firma del contrato',
        'total_contrato' : 'Monto de contrato [COP]',
        'clase' : 'Tipo de producto'
    }
)
fig.show()

# +
#Top 5 de departamentos con mayor monto de contratos por tipo de producto
windowToken = Window.partitionBy('clase').orderBy(col('total_contrato').desc())

secop_depto = secop.\
groupby('clase', 'departamento_entidad').\
agg(sum('valor_contrato_con_adiciones').alias('total_contrato')).\
withColumn('row', row_number().over(windowToken)).\
filter(col('row') <= 5).\
toPandas()

fig = px.bar(
    secop_depto,
    y='departamento_entidad',
    x='total_contrato',
    color='clase',
    facet_col='clase',
    facet_col_wrap=2,
    facet_col_spacing=0.2,
    height=900, width=900,
    orientation='h',
    labels={
        'departamento_entidad' : '',
        'total_contrato' : 'Monto de contrato [COP]',
        'clase' : 'Tipo de producto'
    }
)
fig.update(layout_showlegend=False)
fig.update_xaxes(autorange=True, matches=None, showticklabels=True)
fig.update_yaxes(autorange=True, matches=None, showticklabels=True, categoryorder = 'total ascending')

fig.show()
# +
#Porcentade de contratos con adiciones por tipo de producto
secop_clase_conteo = secop.\
groupby('clase').\
agg(mean('binary_adiciones').alias('per_adiciones')).\
toPandas()

fig = px.bar(
    secop_clase_conteo,
    x='clase',
    y='per_adiciones',
    labels={
        'per_adiciones' : '% de Contratos con adición',
        'clase' : 'Tipo de producto'
    }
)
fig.update_xaxes(categoryorder = 'total descending')
fig.show()
# +
#Porcentade de contratos con adiciones por tipo de producto
secop_clase_conteo = secop.\
groupby('departamento_entidad').\
agg(mean('binary_adiciones').alias('per_adiciones')).\
toPandas()

#Carga del dataframe con geometría
col_deps = gpd.read_file(geo_file_path)
col_deps['Nombre'] = col_deps.loc[:,'Nombre'].apply(text_normalize_cat)
col_deps.loc[col_deps['Nombre'] == 'SAN ANDRES Y PROVIDENCIA','Nombre'] = 'SAN ANDRES'
col_deps.loc[col_deps['Nombre'] == 'GUANIA','Nombre'] = 'GUAINIA'

#Join de dataframes
departamento_total = col_deps.merge(secop_clase_conteo,
                                    how='inner',
                                    left_on = 'Nombre',
                                    right_on = 'departamento_entidad')

fig, ax = plt.subplots(1, figsize=(10, 10))
ax.set_aspect('equal');                     
ax.set_axis_off();

departamento_total.plot(column='per_adiciones',
                        ax=ax,
                        cmap='Blues', linewidth=0.3,
                        scheme='fisher_jenks',
                        legend=True,
                        legend_kwds={'loc': 'lower left'});

col_deps.geometry.boundary.plot(linewidth=0.5, edgecolor='#444444', 
                                color=None, ax=ax);

fig.savefig(os.path.join(fig_path, 'choropleth_per_adiciones.png'))

# +
#Porcentade de contratos con adiciones por tipo de producto
secop_clase_conteo = secop.\
groupby('departamento_entidad').\
agg(sum('valor_contrato_con_adiciones').alias('total_contrato')).\
toPandas()

#Carga del dataframe con geometría
col_deps = gpd.read_file(geo_file_path)
col_deps['Nombre'] = col_deps.loc[:,'Nombre'].apply(text_normalize_cat)
col_deps.loc[col_deps['Nombre'] == 'SAN ANDRES Y PROVIDENCIA','Nombre'] = 'SAN ANDRES'
col_deps.loc[col_deps['Nombre'] == 'GUANIA','Nombre'] = 'GUAINIA'

#Join de dataframes
departamento_total = col_deps.merge(secop_clase_conteo,
                                    how='inner',
                                    left_on = 'Nombre',
                                    right_on = 'departamento_entidad')

fig, ax = plt.subplots(1, figsize=(10, 10))
ax.set_aspect('equal');                     
ax.set_axis_off();

departamento_total.plot(column='total_contrato',
                        ax=ax,
                        cmap='Blues', linewidth=0.3,
                        scheme='fisher_jenks',
                        legend=True,
                        legend_kwds={'loc': 'lower left'});

col_deps.geometry.boundary.plot(linewidth=0.5, edgecolor='#444444', 
                                color=None, ax=ax);

fig.savefig(os.path.join(fig_path, 'choropleth_total_contrato.png'))


# +
#Porcentade de contratos con adiciones por tipo de producto
secop_clase_conteo = secop.\
groupby('departamento_entidad').\
count().\
toPandas()

#Carga del dataframe con geometría
col_deps = gpd.read_file(geo_file_path)
col_deps['Nombre'] = col_deps.loc[:,'Nombre'].apply(text_normalize_cat)
col_deps.loc[col_deps['Nombre'] == 'SAN ANDRES Y PROVIDENCIA','Nombre'] = 'SAN ANDRES'
col_deps.loc[col_deps['Nombre'] == 'GUANIA','Nombre'] = 'GUAINIA'

#Join de dataframes
departamento_total = col_deps.merge(secop_clase_conteo,
                                    how='inner',
                                    left_on = 'Nombre',
                                    right_on = 'departamento_entidad')

fig, ax = plt.subplots(1, figsize=(10, 10))
ax.set_aspect('equal');                     
ax.set_axis_off();

departamento_total.plot(column='count',
                        ax=ax,
                        cmap='Blues', linewidth=0.3,
                        scheme='fisher_jenks',
                        legend=True,
                        legend_kwds={'loc': 'lower left'});

col_deps.geometry.boundary.plot(linewidth=0.5, edgecolor='#444444', 
                                color=None, ax=ax);

fig.savefig(os.path.join(fig_path, 'choropleth_count.png'))
# -

# ## Nube de palabas por tipo de producto

secop_wc = secop.\
select('clase', normalize_text('objeto_del_contrato_a_la_firma').alias('text')).\
groupby('clase').\
agg(concat_ws(' ',collect_list('text')).alias('text')).\
toPandas()

for index, row in secop_wc.iterrows():
    plot_by_doc = wordcloud_graph(
                            text = row['text'],
                            title= row['clase'],
                            stopword_language = 'spanish')
    plot_by_doc.to_file(os.path.join(fig_path, 'wordcloud_{}.png'.format(row['clase'])))

# ## Frecuencia de aparición de palablas por tipo de producto

secop_tokens = count_token_by_doc(df=secop,
                                  document_col='clase',
                                  text_col = 'objeto_del_contrato_a_la_firma')

save_spark(dataframe=secop_tokens,
           lake_path=silver_path,
           file_name=silver_file_name,
           overwriteSchema='true')

secop_tokens = load_data_to_spark(lake_path = silver_path, file_name=silver_file_name)

# ### TF-IDF

# +
#Pipeline tf-idf
secop_tfidf = tfidf_pipeline(df=secop_tokens,
                              document_col='clase',
                              token_col='token')

save_spark(dataframe=secop_tfidf,
           lake_path=silver_path,
           file_name='modelo_tfidf.delta',
           overwriteSchema='true')

# +
#Palabras con mayor peso de tf-idf
fig = plot_top_n_ir(df=secop_tfidf,
                    document_col='clase',
                    ir_col='tf-idf',
                    n=10)
fig.update_layout(autosize=False,width=800, height=1200)
fig.update(layout_showlegend=False)

fig.show()
# -

# ### TF-Entropy

# +
#Pipeline tf-Entropy
secop_tfEnt = tfent_pipeline(df=secop_tokens,
                             document_col='clase',
                             token_col='token')

save_spark(dataframe=secop_tfEnt,
           lake_path=silver_path,
           file_name='modelo_tfent.delta',
           overwriteSchema='true')
# -

#Palabras con mayor peso de tf-logEntropy en cada una de las novelas de Jane Austen 
fig = plot_top_n_ir(df=secop_tfEnt,
                    document_col='clase',
                    ir_col='tf-Entropy',
                    n=10)
fig.update_layout(autosize=False,width=800, height=1200)
fig.update(layout_showlegend=False)
fig.show()

# +
#Distribución de la frecuencia de los términos 
fig = px.histogram(secop_tfidf.toPandas(),
                   x='tf',
                   color="clase",
                   facet_col='clase',
                   facet_col_wrap=2,
                   facet_col_spacing=0.1,
                   height=900, width=800,)

fig.update(layout_showlegend=False)
fig.update_xaxes(range=[0, 0.0009], matches=None, showticklabels=True)
fig.update_yaxes(autorange=True, matches=None, showticklabels=True)
fig.show()
