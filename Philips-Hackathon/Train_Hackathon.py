'''importing the libraries'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# importing the dataset
dataset = pd.read_csv('train.csv')

'''description of the price range'''
description = dataset.describe()

'''information about data'''
dataset.info()

'''Presenting the uniques values'''
unique_outputs = np.unique(dataset.iloc[:, -1].values)

'''heatmap of the dataset'''
sns.heatmap(dataset.corr(), annot = True, linewidths = 0.5)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

''' histogram of the plot'''
dataset.hist()

'''pairplot of the dataset'''
sns.pairplot(dataset)

''' splitting the data in training and test data'''
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.neighbors import KNeighborsClassifier
train_acc = []
test_acc = []

for i in range(1, 11):
    classifier = KNeighborsClassifier(n_neighbors = i)
    classifier.fit(Xtrain, ytrain)
    train_acc.append(classifier.score(Xtrain, ytrain))
    test_acc.append(classifier.score(Xtest, ytest))
    
plt.plot(range(1, 11), train_acc, label = 'train_acc')
plt.plot(range(1, 11), test_acc, label = 'test_acc')
plt.show()    

classifier = KNeighborsClassifier(n_neighbors = 10)
classifier.fit(Xtrain, ytrain)

'''predictinng the results'''
y_pred = classifier.predict(Xtest)

for i, j in zip(ytest, y_pred):
    print(i, j, sep = '==>')

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, y_pred)

accuracy = (124+104+115+130)/500

from sklearn.metrics import r2_score
r2_score(ytest, y_pred)
    

    


    

''' improved method '''

sns.heatmap(dataset.corr(), annot = True, linewidths = 0.5)

X1 = dataset.iloc[:, [0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19]].values
y1 = dataset.iloc[:, -1].values

# splitting up the datasets
Xtrain1, Xtest1, ytrain1, ytest1 = train_test_split(X1, y1, test_size = 0.25, random_state = 0)


train_acc1 = []
test_acc1 = []

for i in range(1, 11):
    classifier1 = KNeighborsClassifier(n_neighbors = i)
    classifier1.fit(Xtrain1, ytrain1)
    train_acc1.append(classifier1.score(Xtrain1, ytrain1))
    test_acc1.append(classifier1.score(Xtest1, ytest1))
    
plt.plot(range(1, 11), train_acc1, label = 'train_acc1')
plt.plot(range(1, 11), test_acc1, label = 'test_acc1')
plt.show()

classifier1 = KNeighborsClassifier(n_neighbors = 10)
classifier1.fit(Xtrain1, ytrain1)

# predicting the new results after removig the columns with negative correaltions
y_pred1 = classifier1.predict(Xtest1)

for i, j, in zip(ytest, y_pred1):
    print(i, j, sep = '==>' )
    
cm1 = confusion_matrix(ytest1, y_pred1)

accuracy_2 = (124+104+115+131)/500

from sklearn.metrics import mean_squared_error
mean_squared_error(ytest1, y_pred1)

'''for test data'''
from sklearn.metrics import precision_recall_fscore_support as score
precision,recall,fscore,support = score(ytest1, y_pred1, average = 'macro')

'''for train data'''
ypred_train = classifier.predict(Xtrain)
from sklearn.metrics import precision_recall_fscore_support as score
precision1,recall1,fscore1,support1 = score(ytrain1, ypred_train, average = 'macro')


''' takin the test dataset'''

dataset2 = pd.read_csv('test.csv')

dataset2.describe()
dataset2.info()
X2 = dataset2.drop(columns = ['id', 'clock_speed', 'mobile_wt', 'touch_screen']).values
output = classifier1.predict(X2)
np.unique(output)

type(output)

df = pd.DataFrame(output)
type(df)

df.to_csv('mobile_price_classification.csv')


