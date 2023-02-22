"""
Created on Wed Nov 30 2022

@author: Syed Misba Shahriyaar
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import seaborn as sns

# Step 1 - Load Data
dataset = pd.read_csv("iphone_purchase_dataset.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values

# Step 2 - Convert Gender to number
labelEncoder_gender =  LabelEncoder()
X[:,0] = labelEncoder_gender.fit_transform(X[:,0])

# Optional - if you want to convert X to float data type
# X = np.vstack(X[:, :]).astype(np.float)


# Step 3 - Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# Step 4 - Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Step 5 - Fit KNN Classifier
# metric = minkowski and p=2 is Euclidean Distance
# metric = minkowski and p=1 is Manhattan Distance
classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski",p=2)
classifier.fit(X_train, y_train)

# Step 6 - Make Prediction
y_pred = classifier.predict(X_test)


# Step 7 - Confusion Matrix
print('Confusion Matrix of KNN Algorithm: ')
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy score:",accuracy)
precision = metrics.precision_score(y_test, y_pred)
print("Precision score:",precision)
recall = metrics.recall_score(y_test, y_pred)
print("Recall score:",recall)

# Step 8 - Make New Predictions
x1 = sc_X.transform([[1,21,40000]])
x2 = sc_X.transform([[1,21,80000]])
x3 = sc_X.transform([[0,21,40000]])
x4 = sc_X.transform([[0,21,80000]])
x5 = sc_X.transform([[1,41,40000]])
x6 = sc_X.transform([[1,51,100000]])
x7 = sc_X.transform([[0,41,40000]])
x8 = sc_X.transform([[0,41,80000]])

print("Male aged 21 making $40k will buy iPhone:", classifier.predict(x1))
print("Male aged 21 making $80k will buy iPhone:", classifier.predict(x2))
print("Female aged 21 making $40k will buy iPhone:", classifier.predict(x3))
print("Female aged 21 making $80k will buy iPhone:", classifier.predict(x4))
print("Male aged 41 making $40k will buy iPhone:", classifier.predict(x5))
print("Male aged 51 making $100k will buy iPhone:", classifier.predict(x6))
print("Female aged 41 making $40k will buy iPhone:", classifier.predict(x7))
print("Female aged 41 making $80k will buy iPhone:", classifier.predict(x8))

# labelEncoder_gender =  LabelEncoder()
# X[:,0] = labelEncoder_gender.fit_transform(X[:,0])
#
# plt.rcParams["figure.figsize"] = [7.00, 3.50]
# plt.rcParams["figure.autolayout"] = True
#
# n_neighbors = 5
# h = .02
#
# cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
# cmap_bold = ['darkorange', 'c', 'darkblue']
#
#
# clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
# clf.fit(X, y)
#
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
# np.arange(y_min, y_max, h))
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
#
# plt.figure()
#
# plt.contourf(xx, yy, Z, cmap=cmap_light)
#
# sns.scatterplot(x=X[:, 0], y=X[:, 1],
# palette=cmap_bold, alpha=1.0, edgecolor="black")
#
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
#
# plt.title("classification (k = %i, 'uniform' = '%s')"
# % (n_neighbors, 'uniform'))
#
# plt.xlabel('Age')
# plt.ylabel('Purchase')
#
# plt.show()