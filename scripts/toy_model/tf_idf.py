import pandas as pd
import plotly.express as px

d = {
    'ID_Documento': ['1', '1', '1', '1', '1', '1', '1', '1', '2', '2', '2', '2', '2', '2', '3', '3', '3', '3', '3', '3'],
    't': ['Alice', 'Bob', 'carro', 'casa', 'tienen', 'un', 'una', 'y', 'Alice', 'carro', 'gato', 'tiene', 'un', 'y', 'Bob', 'carro', 'perro', 'tiene', 'un', 'y'],
    'tfidf':[0.04, 0.04, 0, 0.12, 0.12, 0, 0.12, 0, 0.06, 0, 0.15, 0.06, 0, 0, 0.06, 0, 0.15, 0.06, 0, 0, ]
}

df = pd.DataFrame(d).sort_values(by=['tfidf'])

df

fig = px.bar(
    df,
    x='t',
    y='tfidf',
    color = 'ID_Documento',
    labels={'t':'Termino', 'tfidf':'tf-idf', 'ID_Documento':'Documento'})
fig.show()

df = pd.DataFrame(d)
df

# ?pd.DataFrame
