import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from sklearn import linear_model, neighbors
from sklearn.cluster import KMeans
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error

from processing_methods import read_data, plot_hot_days, \
                               plot_precipitation_means, plot_testset_clusters, \
                               plot_rel_yields, seperate_clusters

# Read in the data:
df = read_data('data/data.xlsx')
df_test = read_data('data/data_test.xlsx')

# Plot the average amount of hot days for each state:
plot_hot_days(df)

# Plot the average precipitation during summer for each state:
plot_precipitation_means(df)

# Plot the yield losses data of every state:
plot_rel_yields(df)

# Define input and target data for the model:
X = np.array(df[['airTemp_mean_summer', 'hot_days_average']].astype(float))
X_train = X[:210]
X_test = X[210:]

y = df['rel_yield_loss'].astype(float)
y_train = y[:210]
y_test = y[210:]

# User: --> Type the feature values for prediction:
X_val = np.array([[19.8, 18]])
X_val = X_val.reshape(1, -1)

# Initialize the regression model:
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
y_pred_val = regr.predict(X_val)

'''
APPLYING K-MEANS CLUSTERING ON TEST DATA
-----------------------------------------
---> seperates the dataset into clusters with mostly positive and negative
     relative yield loss
'''

# Initialize the k-Means-Clustering:
km = KMeans(n_clusters=4)
y_predicted = km.fit_predict(df_test[['airTemp_mean_summer', 'hot_days_average', 'rel_yield_loss']])

# Print the cluster centroids:
centroid_data = km.cluster_centers_
print("================================\n")
print("Cluster Centroids: \n")
print(centroid_data)
print("================================\n")

df_test['cluster'] = y_predicted

# Plot the clustered testdata with the model prediction:
plot_testset_clusters(df_test, X_val, y_pred_val, km)

# Initialize the Squared Error:
k_rng = range(1, 10)
sse = []  # Sum of squared error
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df_test[['airTemp_mean_summer', 'hot_days_average', 'rel_yield_loss']])
    sse.append(km.inertia_)

#print("Sum of squared error (SSE) for different k-values: ", sse)
#print(sse)

# Seperate the testset into the detected clusters:
df1, df2, df3, df4 = seperate_clusters(df_test)

'''
APPLYING K-NEAREST NEIGHBORS ALGORITHM
---------------------------------------
--> assigns predicted crop yield value to the closest cluster in dataset
--> helps to get an insight about whether a predicted crop yield value belongs
    to a cluster determined by negative crop yield (yield loss)
'''

# Declare data for k-Nearest-Neighbors:
X_KNN_train = X_test
X_KNN_test = X_test
y_KNN_train = y_predicted
y_KNN_test = y_predicted

# Initialize k-Nearest-Neighbors:
clf = neighbors.KNeighborsClassifier()
clf.fit(X_KNN_train, y_KNN_train)
example = X_val

# Make the classification with kNN:
prediction = clf.predict(example)
accuracy = clf.score(X_KNN_test, y_KNN_test)

# Print classification and accuracy:
print("=======================================\n")
print("Current input belongs to cluster: ", prediction)
print("=======================================\n")
print("Accuracy of cluster classification: ", accuracy)
print("=======================================\n")
