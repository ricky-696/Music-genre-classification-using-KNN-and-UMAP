import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve,silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE # recursive feature elimination
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

reduced_data = PCA(n_components = 3).fit_transform(data_x) # use PCA do Dimension Reduction
principalDf = pd.DataFrame(data = reduced_data, columns = ['principal component 1', 'principal component 2', 'principal component 3'])

print(principalDf)
X_train, X_test, y_train, y_test = train_test_split(principalDf, data_y, test_size =0.3, random_state=42)

# function of model assess(評估)
def model_assess(model, title = "Default"):
    model.fit(X_train, y_train) # fit model
    preds = model.predict(X_test)
    print('Accuracy', title, ':', round(accuracy_score(y_test, preds), 5), '\n')
    # Confusion Matrix
    confusion_matr = confusion_matrix(y_test, preds) #normalize = 'true'
    plt.figure(figsize = (16, 9))
    sns.heatmap(confusion_matr, cmap="Blues", annot=True, 
        xticklabels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
        yticklabels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"])
    filename = "conf matrix_" + title
    plt.savefig(filename)

# KNN
knn = KNeighborsClassifier(n_neighbors=10)
model_assess(knn, "KNN")