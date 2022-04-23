import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

data = pd.read_csv('Data/features_3_sec.csv')
# remove english letter (regular)
data = data.iloc[0:, 2:] # index locate
data_y = data['label'] # answer
data_x = data.loc[:, data.columns != 'label'] # name locate


#### NORMALIZE X ####
cols = data_x.columns
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(data_x)
data_x = pd.DataFrame(np_scaled, columns = cols)





