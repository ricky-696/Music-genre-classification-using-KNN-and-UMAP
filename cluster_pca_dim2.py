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

reduced_data = PCA(n_components = 3).fit_transform(data_x) # use PCA do Dimension Reduction
principalDf = pd.DataFrame(data = reduced_data, columns = ['principal component 1', 'principal component 2', 'principal component 3'])

# concatenate with target label
finalDf = pd.concat([principalDf, data_y], axis = 1)
print(finalDf)
plt.figure(figsize = (16, 9))
sns.scatterplot(x = "principal component 1", y = "principal component 2", data = finalDf, hue = "label", alpha = 0.7,
               s = 100)

plt.title('PCA on Genres', fontsize = 25)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 10)
plt.xlabel("Principal Component 1", fontsize = 15)
plt.ylabel("Principal Component 2", fontsize = 15)
plt.savefig("PCA Scattert_NORMALIZE_3sec_3Dimension.png")



