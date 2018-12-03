""" RANDOM FOREEST INTITION """
# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# importing the dataset
dataset = pd.read_csv('Data.csv', names = ['Frequency', 'AngleOfAttack', 'ChordLength', 'Velocity', 'thickness', 'soundScale'])
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#applied stadard scaler 

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X = std.fit_transform(X)
y = std.fit_transform(y.reshape(-1,1))

# splitting up the dtasets
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.25, random_state = 0)

sns.heatmap(dataset.corr())

# fitting the model into Random Forest regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(Xtrain, ytrain)

# predicting the values
y_pred = regressor.predict(Xtest)

# printing the accuracies
regressor.score(Xtrain, ytrain)
regressor.score(Xtest, ytest)

from sklearn.metrics import r2_score
print(r2_score(ytest, y_pred))

from sklearn.metrics import mean_squared_error
mean_squared_error(ytest, y_pred)

# visualisingthe result
n_features = dataset.iloc[:, :-1].values.shape[1]
plt.barh(range(n_features), regressor.feature_importances_, align='center', color = 'g')
plt.yticks(np.arange(n_features),('Frequency', 'AngleOfAttack', 'ChordLength', 'Velocity', 'thickness', 'soundScale') )
plt.xlabel('Sound Scale')
plt.ylabel('Feature')
plt.show()


