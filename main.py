from sklearn.metrics import accuracy_score, silhouette_samples, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings(action='ignore')

heart_data= pd.read_csv('defectiveheartdata.csv')
heart_data

heart_data.shape

heart_data.info()

heart_data.describe()


heart_data.isnull().sum()

heart_data.duplicated().sum()

duplicates = heart_data.duplicated()
heart_data[duplicates].sort_values(by = 'age')

heart_data.drop_duplicates(inplace = True)
heart_data.duplicated().sum()

x=heart_data.drop(columns='target',axis=1)
y= heart_data['target']

print(x)

print(y)

x_train, x_test, y_train, y_test=train_test_split(x , y, test_size=0.2, stratify=y,random_state=2)
print (x.shape, x_train.shape, x_test.shape)

classify_model= LogisticRegression()
classify_model.fit(x_train,y_train)

x_train_prediction = classify_model.predict(x_train)
train_accuracy= accuracy_score(x_train_prediction, y_train)
print ('accuracy of train data:',train_accuracy*100,'%')

x_test_prediction = classify_model.predict(x_test)
test_accuracy= accuracy_score(x_test_prediction, y_test)
print ('accuracy of model using test data:',test_accuracy*100,'%')

input_data = (41,0,1,130,204,0,0,172,0,1.4,2,0,2)

input_data_as_numpy_array= np.asarray(input_data)

input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)

prediction = classify_model.predict(input_data_reshaped)
print (prediction)
if (prediction[0]==0):
      print ('The person does not have heart disease')
else:
  print ('The person has heart disease')

plt.scatter(heart_data['age'],heart_data['chol'])

train_heart_data_kmean = KMeans(n_clusters=2)
train_heart_data_kmean.fit(x_train,y_train)
heart_data_cluster = train_heart_data_kmean.labels_
heart_data_cluster

test_heart_data_kmean = KMeans(n_clusters=2)
test_heart_data_kmean.fit(x_test,y_test)
heart_data_cluster = test_heart_data_kmean.labels_
heart_data_cluster

k_rng = range(1, 14)
sse = []
for k in k_rng:
    train_heart_data_kmean = KMeans(n_clusters=k)
    train_heart_data_kmean.fit(x_train,y_train)
    sse.append(train_heart_data_kmean.inertia_)
    plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)

k_rng = range(1, 14)
sse = []
for k in k_rng:
    test_heart_data_kmean = KMeans(n_clusters=k)
    test_heart_data_kmean.fit(x_test,y_test)
    sse.append(test_heart_data_kmean.inertia_)
    plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)
