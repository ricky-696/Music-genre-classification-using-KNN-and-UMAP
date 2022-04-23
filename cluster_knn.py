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

data = pd.read_csv('Data/features_3_sec.csv')
# remove english letter (regular)
data = data.iloc[0:, 2:] # index locate
data_y = data['label'] # answer
print(data_y)
data_x = data.loc[:, data.columns != 'label'] # name locate
print(data_x)


#### NORMALIZE data_x ####
cols = data_x.columns
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(data_x)
data_x = pd.DataFrame(np_scaled, columns = cols)
print(data_x)

# split data into training(70%) and testing(30%)
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size =0.3, random_state=42)

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

