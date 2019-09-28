# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 18:54:37 2019

@author: Igor
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def _convert_str_to_int(x):
    if x == 'yes':
        return 1.0
    elif x == 'no':
        return 0.0


# Importing the dataset
# dataset downloaded from https://www.kaggle.com/dipam7/student-grade-prediction#student-mat.csv
dataset = pd.read_csv('student-mat.csv')

# Data pre-processing
# Criteria defined in dataset related article
dataset['approved'] = dataset['G3'] >= 10
dataset = dataset.drop(columns=['school', 'G1', 'G2', 'G3'])

convert_str_to_int = np.vectorize(_convert_str_to_int)

dataset['schoolsup'] = convert_str_to_int(dataset['schoolsup'])
dataset['famsup'] = convert_str_to_int(dataset['famsup'])
dataset['paid'] = convert_str_to_int(dataset['paid'])
dataset['activities'] = convert_str_to_int(dataset['activities'])
dataset['nursery'] = convert_str_to_int(dataset['nursery'])
dataset['higher'] = convert_str_to_int(dataset['higher'])
dataset['internet'] = convert_str_to_int(dataset['internet'])
dataset['romantic'] = convert_str_to_int(dataset['romantic'])


X = dataset.iloc[:, :29].values
y = dataset.iloc[:, -1].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])
labelencoder_X_4 = LabelEncoder()
X[:, 4] = labelencoder_X_4.fit_transform(X[:, 4])
labelencoder_X_7 = LabelEncoder()
X[:, 7] = labelencoder_X_7.fit_transform(X[:, 7])
labelencoder_X_8 = LabelEncoder()
X[:, 8] = labelencoder_X_8.fit_transform(X[:, 8])
labelencoder_X_9 = LabelEncoder()
X[:, 9] = labelencoder_X_9.fit_transform(X[:, 9])
labelencoder_X_10 = LabelEncoder()
X[:, 10] = labelencoder_X_10.fit_transform(X[:, 10])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# End Pre-processing
#------------------------------------------------------------------------------

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm_precision = ((cm[0][0] + cm[1][1])/len(y_test) ) * 100


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
mean_accuracy = accuracies.mean()
standard_deviation_accuracies = accuracies.std()

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

random_student = [X_test[-1]]
random_student_prediction = classifier.predict(random_student)
print('Will the student pass the class? {}\nDid we predict it correctly? {}'
      .format('Yes' if random_student_prediction[0] else 'No',
              'Yes' if random_student_prediction[0] == y_test[-1] else 'No'))
print("Actual student's attributes:\n{}".format(dataset.tail(n=1)))