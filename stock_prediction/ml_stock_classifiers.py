# -*- coding: utf-8 -*-
"""

@author: Igor
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def _calculate_result(open_value, close_value):
    # Return 1 if stock value increased and 0 if drecreased.
    # This dataset does not contain data that maintained the same value
    if open_value > close_value:
        return 1
    return 0
   

# Importing the dataset
# dataset downloaded from https://www.kaggle.com/aaron7sun/stocknews#DJIA_table.csv
dataset = pd.read_csv('DJIA_table.csv')


#Data pre-processing

X = dataset
X = X[['Open', 'Close', 'Volume']]
X['Variance Percentage'] = ((X['Close'] - X['Open']) / X['Open']) * 100
X['Variance Percentage'] = X['Variance Percentage'].round(2)

Cluster_set = X[['Volume', 'Variance Percentage']]
calculate_result = np.vectorize(_calculate_result)
y = calculate_result(X['Open'], X['Close'])


X = X[['Open', 'Volume']]
X = X.iloc[:, :].values
Cluster_set = Cluster_set.iloc[:, :].values


# Clustering algorithm

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(Cluster_set)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(Cluster_set)

# Visualising the clusters
plt.scatter(Cluster_set[y_kmeans == 0, 0], Cluster_set[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Low Volume, Low Variance')
plt.scatter(Cluster_set[y_kmeans == 1, 0], Cluster_set[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'High Volume, High Variance')
plt.scatter(Cluster_set[y_kmeans == 2, 0], Cluster_set[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Average')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters')
plt.xlabel('Volume Traded(mi)')
plt.ylabel('Price variance (%)')
plt.legend()
plt.show()

# End Clustering
# ----------------------------------------------------------------------------

# Classification algorithm (Naive Bayes)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# End Classification
# -----------------------------------------------------------------------------

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
precision = ((cm[0][0] + cm[1][1])/len(y_test) ) * 100

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 1000),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 1000))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Training set)')
plt.xlabel('Stock Price (Open)')
plt.ylabel('Volume Traded(mi)')
plt.legend()
plt.show()

# Visualising the Test set results
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 1000),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 1000))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Training set)')
plt.xlabel('Stock Price (Open)')
plt.ylabel('Volume Traded(mi)')
plt.legend()
plt.show()


random_stock_day = [X_test[-1]]
random_stock_day_prediction = classifier.predict(random_stock_day)
print('Will the stock increase today? {}\nDid we predict it correctly? {}'
      .format('Yes' if random_stock_day_prediction[0] else 'No',
              'Yes' if random_stock_day_prediction[0] == y_test[-1] else 'No'))
