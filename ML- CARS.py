#!/usr/bin/env python
# coding: utf-8

# In[267]:


#Setting It Up
#I collected all of the data above and combined them into one dataframe. The code and details are located here. One challenge was the periodicity of the various features. Our exchange data is daily, some data is monthly, and others quarterly. For our daily exchange rates, I took the last value of each month. For the quarterly data, I copied the quarterly value to each month in that quarter. This gives us a dataframe of monthly data that is easier to work with.
#First, we will import the libraries we will be using and also load our data into a Pandas dataframe.

# Import needed libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import seaborn as sn
import sklearn

# Python magic to show plots inline in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
import datetime as dt
from datetime import datetime
import math


# In[17]:


# Import data
df = pd.read_csv("C:/Users/Kingsley/Desktop/allcars.csv")


# In[18]:


print(df)


# In[19]:


#It is a good practice to understand the data first and try to gather as many insights from it. 
#EDA is all about making sense of data in hand,before getting them dirty with it.

#1. Check for Missing Data
#2. Heatmap for Data Structure
#3. Correlations

df.head()


# In[25]:


df.info()


# In[20]:


#Check for Missing Data
df.isnull().values.any()


# In[21]:


# Show rows where any cell has a NaN
df[df.isnull().any(axis=1)].shape


# In[22]:


#### Drop cells with NaN
df = df.dropna(axis=0,subset=['cTemp'])
df = df.dropna(axis=0,subset=['dtc'])
df = df.dropna(axis=0,subset=['iat'])
df = df.dropna(axis=0,subset=['imap'])
df = df.dropna(axis=0,subset=['tAdv'])


# In[23]:


# Show rows where any cell has a NaN
df[df.isnull().any(axis=1)].shape


# In[24]:


df.describe()


# In[26]:


# Seaborn doesn't handle NaN values, so we can fill them with 0 for now.
df = df.fillna(value=0)

# Pair grid of key variables.
g = sns.PairGrid(df)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
plt.subplots_adjust(top=0.95)
g.fig.suptitle('Pairwise Grid of Numeric Features');


# In[37]:


#To use linear regression for modelling,its necessary to remove correlated variables to improve your model.
#One can find correlations using pandas “.corr()” function and can visualize the correlation matrix using a heatmap in seaborn.

corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, cmap='Blues')
plt.title('Correlation Heatmap of Numeric Features')

#The very light and very dark boxes show a strong positive or negative correlation between the features.


# In[191]:


#Select variables with complete dataset (no nan or zero)
df1 = pd.DataFrame(df,columns=['tripID','gps_speed','cTemp','eLoad','iat','imap','rpm','speed'])


# In[192]:


#Remove correlated variables before feature selection.
corrMatrix = df1.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

#Here, it can be infered that IAT – In-dash automatic transmission “iat” has strong positive correlation with circular temperature “cTemp”


# In[269]:


#Make final dataset
df1.columns


# In[210]:


cols = df1.columns.tolist()


# In[195]:


df1.to_csv('allcars.csv')


# In[196]:


df1.dtypes


# In[197]:


# FEATURE ENGINEERING
# Define custom function to create lag values
def feature_lag(features):
    for feature in features:
        df[feature + '-lag1'] = df[feature].shift(1)
        df[feature + '-lag2'] = df[feature].shift(2)
        df[feature + '-lag3'] = df[feature].shift(3)
        df[feature + '-lag4'] = df[feature].shift(4)

# Define columns to create lags for
features = ['tripID','gps_speed','cTemp','eLoad','iat','imap','rpm','speed']

# Call custom function
feature_lag(features)


# In[198]:


#predict gps speed 3, 6, and 12 months ahead.

df1['y3'] = df.gps_speed.shift(-3)
df1['y6'] = df.gps_speed.shift(-6)
df1['y12'] = df.gps_speed.shift(-12)


# In[212]:


df1 = df1.dropna(axis=0,subset=['y3'])
df1 = df1.dropna(axis=0,subset=['y6'])
df1 = df1.dropna(axis=0,subset=['y12'])


# In[213]:


df1.dtypes


# In[216]:


#Split into Training and Test Data
#Cross validation is always desired when training machine learning models to be able to trust the generality of the model created. We will split our data into training and test data using Scikit learn's built in tools. Also for scikit learn we need to separate our dataset into inputs and the feature being predicted (or X's and y's).

y = df1['gps_speed']


# In[217]:


X = df1.drop(['gps_speed'], axis=1)


# In[218]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1234)


# In[219]:


X_train.shape, y_train.shape


# In[220]:


X_test.shape, y_test.shape


# In[221]:


X.columns


# In[206]:


df1.dtypes


# In[222]:


from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
# Create linear regression object
regr = LinearRegression()


# In[223]:


# Train the model using the training sets
regr.fit(X_train, y_train)


# In[225]:


# Make predictions using the testing set
lin_pred = regr.predict(X_test)


# In[226]:


linear_regression_score = regr.score(X_test, y_test)
linear_regression_score


# In[227]:


linear_regression_score = regr.score(X_train, y_train)
linear_regression_score


# In[228]:


from math import sqrt
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, lin_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, lin_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, lin_pred))


# In[229]:


plt.scatter(y_test, lin_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Linear Regression Predicted vs Actual')
plt.show()


# In[230]:


### Neural Network Regression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create MLPRegressor object
mlp = MLPRegressor()


# In[231]:


# Train the model using the training sets
mlp.fit(X_train, y_train)


# In[232]:


# Score the model
neural_network_regression_score = mlp.score(X_test, y_test)
neural_network_regression_score


# In[233]:


# Score the model
neural_network_regression_score = mlp.score(X_train, y_train)
neural_network_regression_score


# In[234]:


# Make predictions using the testing set
nnr_pred = mlp.predict(X_test)


# In[235]:


# The mean squared error
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, nnr_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, nnr_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, nnr_pred))


# In[236]:


plt.scatter(y_test, nnr_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Neural Network Regression Predicted vs Actual')
plt.show()


# In[237]:


###Lasso
from sklearn.linear_model import Lasso

lasso = Lasso()


# In[238]:


lasso.fit(X_train, y_train)


# In[239]:


# Score the model
lasso_score = lasso.score(X_test, y_test)
lasso_score


# In[240]:


# Score the model
lasso_score = lasso.score(X_train, y_train)
lasso_score


# In[241]:


# Make predictions using the testing set
lasso_pred = lasso.predict(X_test)


# In[242]:


print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, lasso_pred)))

# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, lasso_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, lasso_pred))


# In[243]:


plt.scatter(y_test, lasso_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Lasso Predicted vs Actual')
plt.show()


# In[244]:


##ElasticNet
from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet()
elasticnet.fit(X_train, y_train)


# In[245]:


elasticnet_score = elasticnet.score(X_test, y_test)
elasticnet_score


# In[246]:


elasticnet_score = elasticnet.score(X_test, y_test)
elasticnet_score


# In[247]:


elasticnet_pred = elasticnet.predict(X_test)


# In[248]:


# The mean squared error
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, elasticnet_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, elasticnet_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, elasticnet_pred))


# In[249]:


###Decision Forest Regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create Random Forrest Regressor object
regr_rf = RandomForestRegressor(n_estimators=200, random_state=1234)


# In[250]:


# Train the model using the training sets
regr_rf.fit(X_train, y_train)


# In[251]:


regr_rf.fit(X_test, y_test)


# In[252]:


# Score the model
decision_forest_score = regr_rf.score(X_test, y_test)
decision_forest_score


# In[253]:


# Make predictions using the testing set
regr_rf_pred = regr_rf.predict(X_test)


# In[254]:


from math import sqrt
# The mean squared error
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, regr_rf_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, regr_rf_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, regr_rf_pred))


# In[255]:


features = X.columns
importances = regr_rf.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()


# In[256]:


plt.scatter(y_test, regr_rf_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Decision Forest Predicted vs Actual')
plt.show()


# In[257]:


#Extra Trees Regression

from sklearn.ensemble import ExtraTreesRegressor

extra_tree = ExtraTreesRegressor(n_estimators=200, random_state=1234)


# In[258]:


extra_tree.fit(X_train, y_train)


# In[259]:


extratree_score = extra_tree.score(X_test, y_test)
extratree_score


# In[260]:


extratree_score = extra_tree.score(X_train, y_train)
extratree_score


# In[261]:


extratree_pred = extra_tree.predict(X_test)


# In[262]:


print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, extratree_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, extratree_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, extratree_pred))


# In[263]:


features = X.columns
importances = extra_tree.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()


# In[264]:


plt.scatter(y_test, extratree_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Extra Trees Predicted vs Actual')
plt.show()


# In[266]:


#Evaluate Models
print("Scores:")
print("Linear regression score: ", linear_regression_score)
print("Neural network regression score: ", neural_network_regression_score)
print("Lasso regression score: ", lasso_score)
print("ElasticNet regression score: ", elasticnet_score)
print("Decision forest score: ", decision_forest_score)
print("Extra Trees score: ", extratree_score)
print("\n")
print("RMSE:")
print("Linear regression RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, lin_pred)))
print("Neural network RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, nnr_pred)))
print("Lasso RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, lasso_pred)))
print("ElasticNet RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, elasticnet_pred)))
print("Decision forest RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, regr_rf_pred)))
print("Extra Trees RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, extratree_pred)))


# In[ ]:





# In[ ]:




