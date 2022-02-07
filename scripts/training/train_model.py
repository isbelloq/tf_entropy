# -*- coding: utf-8 -*-
# +
from tf_entropy.environment.base import get_data_paths
from tf_entropy.environment.spark import create_spark_session
from tf_entropy.database.io import load_data_to_spark, save_spark
from tf_entropy.datamodels.datadic import SparkDataFrameDic
from tf_entropy.preprocessing.text import spark_token_counter, normalize_text
from tf_entropy.models.feature_extraction import dataframe_score, create_map_model, features_lables_split

from pyspark.sql.functions import explode

from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
import plotly.express as px

import os

spark = create_spark_session()


# -

def list_confusion_matrix(cm,classes):
  df = pd.DataFrame(data = cm,
                    index = pd.MultiIndex.from_product([['Valor real'], classes]),
                    columns = pd.MultiIndex.from_product([['Valor predicho'], classes]))
  
  return df


# # Carga de datos

# +
#Creación de diccionario de datos
secop_entrenamiento = SparkDataFrameDic(
    float_columns = ['valor_contrato_con_adiciones', 'score'],
    categorical_columns = ['nivel_entidad', 'regimen_de_contratacion',
                           'origen_de_los_recursos', 'clase', 'departamento_entidad'],
    text_columns = ['objeto_del_contrato_a_la_firma'])

#Carga de paths del lago de datos
datalake = get_data_paths()

#Definición de path de datos
silver_file_path = os.path.join(datalake.silver_path, 'datos_abiertos/secop/frecuencia_palabras')

secop_file_name = 'secop_entrenamiento.delta'
tfidf_file_name = 'modelo_tfidf.delta'
tfent_file_name = 'modelo_tfent.delta'

#Carga de datos de entrenamiento
secop = load_data_to_spark(
        lake_path=silver_file_path,
        file_name=secop_file_name)

tfidf = load_data_to_spark(
        lake_path=silver_file_path,
        file_name=tfidf_file_name)

tfent = load_data_to_spark(
        lake_path=silver_file_path,
        file_name=tfent_file_name)
# -

# # Preprocesamiento de datos

# +
# #Mapeo de tfidf
# tfidf_map = create_map_model(df=tfidf,
#                             document_col='clase',
#                             token_col='token',
#                             model_col='tf-idf')

# #Mapeo de tfent
# tfent_map = create_map_model(df=tfent,
#                             document_col='clase',
#                             token_col='token',
#                             model_col='tf-entropy')

# +
# text_col = secop_entrenamiento.text_columns[0]

# #Calculo de score con tfidf
# tfidf_model = dataframe_score(df=secop,
#                              df_map=tfidf_map,
#                              id_col='uid',
#                              document_col='clase',
#                              text_col=text_col)

# save_spark(dataframe=tfidf_model,
#            lake_path=silver_path,
#            file_name='secop_tfidfscore.delta',
#            overwriteSchema='true')

# #Calculo de score con tfent
# tfent_model = dataframe_score(df=secop,
#                              df_map=tfent_map,
#                              id_col='uid',
#                              document_col='clase',
#                              text_col=text_col)

# save_spark(dataframe=tfent_model,
#            lake_path=silver_path,
#            file_name='secop_tfentscore.delta',
#            overwriteSchema='true')

# +
tfidf_model = load_data_to_spark(
        lake_path=silver_file_path,
        file_name='secop_tfidfscore.delta')

tfent_model = load_data_to_spark(
        lake_path=silver_file_path,
        file_name='secop_tfentscore.delta')
# -

#Creación de conjutno de datos para modelo
secop_tfidf = secop.join(tfidf_model,on=['uid'],how='inner').distinct().toPandas()
secop_tfidf['score'] = secop_tfidf['score']+1
secop_tfent = secop.join(tfent_model,on=['uid'],how='inner').distinct().toPandas()
secop_tfent['score'] = secop_tfent['score']+1

import plotly.express as px
fig = px.scatter(secop_tfidf,
                 x="valor_contrato_con_adiciones",
                 y="score",
                 color='binary_adiciones',
                 log_x=True,
                 log_y=True,
                 labels={
                     'valor_contrato_con_adiciones' : 'Valor del contrato con adiciones',
                     'score' : 'Score con base en tf-idf',
                     'binary_adiciones' : 'Contrato con adiciones'})
fig.show()

fig = px.scatter(secop_tfent,
                 x="valor_contrato_con_adiciones",
                 y="score",
                 color='binary_adiciones',
                 log_x=True,
                 log_y=True,
                 labels={
                     'valor_contrato_con_adiciones' : 'Valor del contrato con adiciones',
                     'score' : 'Score con base en tf-entropy',
                     'binary_adiciones' : 'Contrato con adiciones'})
fig.show()

# ## Preparación para el entrenamiento

# +
#Pipeline de transformación de columnas
LogTransformer = FunctionTransformer(np.log10)
tf = ColumnTransformer([('onehot', OneHotEncoder(), secop_entrenamiento.categorical_columns),
                        ('log10', LogTransformer, secop_entrenamiento.float_columns)])

tf_ref = ColumnTransformer([('onehot', OneHotEncoder(), secop_entrenamiento.categorical_columns),
                        ('log10', LogTransformer, ['valor_contrato_con_adiciones'])])

# +
#Separación de features y labels
secop_ref_X,secop_ref_y = features_lables_split(df=secop.toPandas(),
                                                y_label='binary_adiciones') 

secop_tfidf_X,secop_tfidf_y = features_lables_split(df=secop_tfidf,
                                                    y_label='binary_adiciones') 

secop_tfent_X,secop_tfent_y= features_lables_split(df=secop_tfent,
                                                    y_label='binary_adiciones')

#Transformación del conjunto de features
X_preprocessed_ref = tf_ref.fit_transform(secop_ref_X)
X_preprocessed_tfidf = tf.fit_transform(secop_tfidf_X)
X_preprocessed_tfent = tf.fit_transform(secop_tfent_X)

#Split tain/test
X_ref_train, X_ref_test, y_ref_train, y_ref_test = train_test_split(X_preprocessed_ref,
                                                                    secop_ref_y,
                                                                    test_size=0.3,     # Proporción de datos usados para el grupo de evaluación.
                                                                    random_state=202202, # Semilla aleatoria para la replicabilidad.
                                                                    stratify=secop_ref_y)   # Estratificar con respecto a la etiqueta.


X_tfidf_train, X_tfidf_test, y_tfidf_train, y_tfidf_test = train_test_split(X_preprocessed_tfidf,
                                                                            secop_tfidf_y,
                                                                            test_size=0.3,     # Proporción de datos usados para el grupo de evaluación.
                                                                            random_state=202202, # Semilla aleatoria para la replicabilidad.
                                                                            stratify=secop_tfidf_y)   # Estratificar con respecto a la etiqueta.

X_tfent_train, X_tfent_test, y_tfent_train, y_tfent_test = train_test_split(X_preprocessed_tfent,
                                                                            secop_tfidf_y,
                                                                            test_size=0.3,     # Proporción de datos usados para el grupo de evaluación.
                                                                            random_state=202202, # Semilla aleatoria para la replicabilidad.
                                                                            stratify=secop_tfent_y)   # Estratificar con respecto a la etiqueta.
# -

# ## Entranamiento

# ### SVM

# +
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer

f1 = make_scorer(f1_score , average='macro')
param_grid = {'gamma': [2**i for i in range(7, 9, 1)],
              'C':     [10**i for i in range(0, 3, 1)],
              'kernel':['rbf']}


grid_clf_ref = GridSearchCV(SVC(),
                            param_grid=param_grid,
                            verbose=1,
                            cv=2,
                            scoring=f1,
                            return_train_score=True)

grid_clf_idf = GridSearchCV(SVC(),
                            param_grid=param_grid,
                            verbose=1,
                            cv=2,
                            scoring=f1,
                            return_train_score=True)

grid_clf_ent = GridSearchCV(SVC(),
                            param_grid=param_grid,
                            verbose=1,
                            cv=2,
                            scoring=f1,
                            return_train_score=True)
# -

# ## Modelo de referencia

# %%time
grid_clf_ref.fit(X_ref_train, y_ref_train)

print(grid_clf_ref.best_params_)

y_pred_ref = grid_clf_ref.predict(X_ref_test)

# +
print('PARA 1\n')
print(f'Precisión: {metrics.precision_score(y_ref_test, y_pred_ref, pos_label=1)}')
print(f'Recall:    {metrics.recall_score(y_ref_test, y_pred_ref, pos_label=1)}')
print(f'F_1 score: {metrics.f1_score(y_ref_test, y_pred_ref, pos_label=1)}')

print('PARA 0\n')
print(f'Precisión: {metrics.precision_score(y_ref_test, y_pred_ref, pos_label=0)}')
print(f'Recall:    {metrics.recall_score(y_ref_test, y_pred_ref, pos_label=0)}')
print(f'F_1 score: {metrics.f1_score(y_ref_test, y_pred_ref, pos_label=0)}')
# -

cnf_matrix_ref = confusion_matrix(y_ref_test, y_pred_ref)
list_confusion_matrix(cnf_matrix_ref, ['0','1'])

# ## Modelo de tfidf

# %%time
grid_clf_idf.fit(X_tfidf_train, y_tfidf_train)

print(grid_clf_idf.best_params_)

y_pred_tfidf = grid_clf_idf.predict(X_tfidf_test)

# +
print('PARA 1\n')
print(f'Precisión: {metrics.precision_score(y_tfidf_test, y_pred_tfidf, pos_label=1)}')
print(f'Recall:    {metrics.recall_score(y_tfidf_test, y_pred_tfidf, pos_label=1)}')
print(f'F_1 score: {metrics.f1_score(y_tfidf_test, y_pred_tfidf, pos_label=1)}')

print('PARA 0\n')
print(f'Precisión: {metrics.precision_score(y_tfidf_test, y_pred_tfidf, pos_label=0)}')
print(f'Recall:    {metrics.recall_score(y_tfidf_test, y_pred_tfidf, pos_label=0)}')
print(f'F_1 score: {metrics.f1_score(y_tfidf_test, y_pred_tfidf, pos_label=0)}')
# -

cnf_matrix_idf = confusion_matrix(y_tfidf_test, y_pred_tfidf)
list_confusion_matrix(cnf_matrix_idf, ['0','1'])

# ## Modelo de tfent

# %%time
grid_clf_ent.fit(X_tfent_train, y_tfent_train)

print(grid_clf_ent.best_params_)

y_pred_tfent = grid_clf_ent.predict(X_tfent_test)

# +
print('PARA 1\n')
print(f'Precisión: {metrics.precision_score(y_tfent_test, y_pred_tfent, pos_label=1)}')
print(f'Recall:    {metrics.recall_score(y_tfent_test, y_pred_tfent, pos_label=1)}')
print(f'F_1 score: {metrics.f1_score(y_tfent_test, y_pred_tfent, pos_label=1)}')

print('PARA 0\n')
print(f'Precisión: {metrics.precision_score(y_tfent_test, y_pred_tfent, pos_label=0)}')
print(f'Recall:    {metrics.recall_score(y_tfent_test, y_pred_tfent, pos_label=0)}')
print(f'F_1 score: {metrics.f1_score(y_tfent_test, y_pred_tfent, pos_label=0)}')
# -

cnf_matrix_ent = confusion_matrix(y_tfent_test, y_pred_tfent)
list_confusion_matrix(cnf_matrix_ent, ['0','1'])
