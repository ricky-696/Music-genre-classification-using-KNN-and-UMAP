import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

color_scheme = {
 'classical': '#FE88FC',
 'jazz': '#F246FE',
 'blues': '#BF1CFD', 
 'metal': '#6ECE58',
 'rock': '#35B779',
 'disco': '#1F9E89',
 'pop': '#Fb9B06',
 'reggae': '#ED6925',
 'hiphop': '#CF4446',   
 'country': '#000004',   
 }

data_o = pd.read_csv('Data/features_3_sec.csv')
# remove english letter (regular)
data = data_o.iloc[0:, 2:] # index locate
data_y = data['label'] # answer
data_x = data.loc[:, data.columns != 'label'] # name locate


#### NORMALIZE X ####
cols = data_x.columns
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(data_x)
data_x = pd.DataFrame(np_scaled, columns = cols)

# use PCA do Dimension Reduction
reduced_data = PCA(n_components = 3).fit_transform(data_x) 
principalDf = pd.DataFrame(data = reduced_data, columns = ['principal component 1', 'principal component 2', 'principal component 3'])



principalDf.columns = ['pc1','pc2','pc3']
principalDf['genre'] = data.iloc[:,-1]
print(principalDf)

principalDf['text'] = data_o[['filename','label']].apply(lambda x: f'{x[0]}<br>{x[1]}', axis=1)

import plotly.graph_objects as go

data = []
for g in principalDf.genre.unique():
    trace = go.Scatter3d(
    x = principalDf[principalDf.genre==g].pc1.values,
    y = principalDf[principalDf.genre==g].pc2.values,
    z = principalDf[principalDf.genre==g].pc3.values,
    mode='markers',
    text = principalDf[principalDf.genre==g].text.values,
    hoverinfo = 'text',
    name=g,
    marker=dict(
            size=3,
            color = color_scheme[g],                
            opacity=0.9,
        )
    )
    data.append(trace)
fig = go.Figure(data=data)

fig.update_layout(title=f'PCA', autosize=False,
                      width=600, height=600,
                      margin=dict(l=50, r=50, b=50, t=50),
                      scene=dict(xaxis=dict(title='pc1'), yaxis=dict(title='pc2'), zaxis=dict(title='pc3'))
                     )
fig.show()
