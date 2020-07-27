#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[62]:


# Import data
df = pd.read_csv("https://raw.githubusercontent.com/Preetinsights/Telematic-OBS-Gear-Prediction/master/allcars.csv")


# In[63]:


print(df)


# In[64]:


#It is a good practice to understand the data first and try to gather as many insights from it. 
#EDA is all about making sense of data in hand,before getting them dirty with it.

#1. Check for Missing Data
#2. Heatmap & Data Structure
#3. Correlations
#4. Uncover a parsimonious model, one which explains the data with a minimum number of predictor variables.
df.head()


# In[65]:


df.info()


# In[18]:


#Check for Missing Data
df.isnull().values.any()


# In[66]:


# Show rows where any cell has a NaN
df[df.isnull().any(axis=1)].shape


# In[67]:


#### Drop cells with NaN
df = df.dropna(axis=0,subset=['cTemp'])
df = df.dropna(axis=0,subset=['dtc'])
df = df.dropna(axis=0,subset=['iat'])
df = df.dropna(axis=0,subset=['imap'])
df = df.dropna(axis=0,subset=['tAdv'])


# In[68]:


# Show rows where any cell has a NaN
df[df.isnull().any(axis=1)].shape


# In[69]:


df.describe()


# In[70]:


# Seaborn doesn't handle NaN values, so we can fill them with 0 for now.
df = df.fillna(value=0)
# Pair grid of key variables.
g = sns.PairGrid(df)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
plt.subplots_adjust(top=0.95)
g.fig.suptitle('Pairwise Grid of Numeric Features');


# In[71]:


g = sns.PairGrid(df, vars=["gps_speed", "speed"])
g = g.map(plt.scatter)


# In[72]:


g = sns.PairGrid(df, vars=["iat", "rpm"])
g = g.map(plt.scatter)
#iat is in-board automatic transmission
#rpm = revolution per minute


# In[73]:


#To use linear regression for modelling,its necessary to remove correlated variables to improve your model.
#One can find correlations using pandas “.corr()” function and can visualize the correlation matrix using a heatmap in seaborn.

corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, cmap='Blues')
plt.title('Correlation Heatmap of Numeric Features')


# In[74]:


#Select variables with complete dataset (no nan or zero)
df1 = pd.DataFrame(df,columns=['tripID','gps_speed','cTemp','eLoad','iat','imap','rpm','speed'])


# In[75]:


#Remove correlated variables before feature selection.
corrMatrix = df1.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

#Here, it can be infered that IAT – In-dash automatic transmission “iat” has strong positive correlation with circular temperature “cTemp”


# In[76]:


#Make final dataset
df.columns


# In[77]:


cols = df.columns.tolist()


# In[78]:


df1.to_csv('allcars.csv')


# In[79]:


df1.dtypes


# In[80]:


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


# In[81]:


#predict gps speed 3, 6, and 12 months ahead.

df1['y3'] = df.gps_speed.shift(-3)
df1['y6'] = df.gps_speed.shift(-6)
df1['y12'] = df.gps_speed.shift(-12)


# In[82]:


df1 = df1.dropna(axis=0,subset=['y3'])
df1 = df1.dropna(axis=0,subset=['y6'])
df1 = df1.dropna(axis=0,subset=['y12'])


# In[83]:


df1.dtypes


# In[84]:


#Split into Training and Test Data
#Cross validation is always desired when training machine learning models to be able to trust the generality of the model created. We will split our data into training and test data using Scikit learn's built in tools. Also for scikit learn we need to separate our dataset into inputs and the feature being predicted (or X's and y's).

y = df1['gps_speed']


# In[85]:


X = df1.drop(['gps_speed'], axis=1)


# In[86]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1234)


# In[87]:


X_train.shape, y_train.shape


# In[88]:


X_test.shape, y_test.shape


# In[89]:


X.columns


# In[90]:


df1.dtypes


# In[91]:


from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
# Create linear regression object
regr = LinearRegression()


# In[92]:


# Train the model using the training sets
regr.fit(X_train, y_train)


# In[93]:


# Make predictions using the testing set
lin_pred = regr.predict(X_test)


# In[94]:


linear_regression_score = regr.score(X_test, y_test)
linear_regression_score


# In[95]:


linear_regression_score = regr.score(X_train, y_train)
linear_regression_score


# In[96]:


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


# In[97]:


plt.scatter(y_test, lin_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Linear Regression Predicted vs Actual')
plt.show()


# In[98]:


### Neural Network Regression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create MLPRegressor object
mlp = MLPRegressor()


# In[99]:


# Train the model using the training sets
mlp.fit(X_train, y_train)


# In[100]:


# Score the model
neural_network_regression_score = mlp.score(X_test, y_test)
neural_network_regression_score


# In[101]:


# Score the model
neural_network_regression_score = mlp.score(X_train, y_train)
neural_network_regression_score


# In[102]:


# Make predictions using the testing set
nnr_pred = mlp.predict(X_test)


# In[103]:


# The mean squared error
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, nnr_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, nnr_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, nnr_pred))


# In[104]:


plt.scatter(y_test, nnr_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Neural Network Regression Predicted vs Actual')
plt.show()


# In[105]:


###Lasso
from sklearn.linear_model import Lasso

lasso = Lasso()


# In[106]:


lasso.fit(X_train, y_train)


# In[107]:


# Score the model
lasso_score = lasso.score(X_test, y_test)
lasso_score


# In[108]:


# Score the model
lasso_score = lasso.score(X_train, y_train)
lasso_score


# In[109]:


# Make predictions using the testing set
lasso_pred = lasso.predict(X_test)


# In[110]:


print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, lasso_pred)))

# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, lasso_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, lasso_pred))


# In[111]:


plt.scatter(y_test, lasso_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Lasso Predicted vs Actual')
plt.show()


# In[112]:


##ElasticNet
from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet()
elasticnet.fit(X_train, y_train)


# In[113]:


elasticnet_score = elasticnet.score(X_test, y_test)
elasticnet_score


# In[114]:


elasticnet_score = elasticnet.score(X_test, y_test)
elasticnet_score


# In[115]:


elasticnet_pred = elasticnet.predict(X_test)


# In[116]:


# The mean squared error
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, elasticnet_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, elasticnet_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, elasticnet_pred))


# In[117]:


###Decision Forest Regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create Random Forrest Regressor object
regr_rf = RandomForestRegressor(n_estimators=200, random_state=1234)


# In[118]:


# Train the model using the training sets
regr_rf.fit(X_train, y_train)


# In[119]:


regr_rf.fit(X_test, y_test)


# In[120]:


# Score the model
decision_forest_score = regr_rf.score(X_test, y_test)
decision_forest_score


# In[121]:


# Make predictions using the testing set
regr_rf_pred = regr_rf.predict(X_test)


# In[122]:


from math import sqrt
# The mean squared error
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, regr_rf_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, regr_rf_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, regr_rf_pred))


# In[123]:


features = X.columns
importances = regr_rf.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()


# In[124]:


plt.scatter(y_test, regr_rf_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Decision Forest Predicted vs Actual')
plt.show()


# In[125]:


#Extra Trees Regression

from sklearn.ensemble import ExtraTreesRegressor

extra_tree = ExtraTreesRegressor(n_estimators=200, random_state=1234)


# In[126]:


extra_tree.fit(X_train, y_train)


# In[127]:


extratree_score = extra_tree.score(X_test, y_test)
extratree_score


# In[128]:


extratree_score = extra_tree.score(X_train, y_train)
extratree_score


# In[129]:


extratree_pred = extra_tree.predict(X_test)


# In[130]:


print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, extratree_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, extratree_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, extratree_pred))


# In[131]:


features = X.columns
importances = extra_tree.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()


# In[132]:


plt.scatter(y_test, extratree_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Extra Trees Predicted vs Actual')
plt.show()


# In[133]:


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


# In[134]:


from IPython.display import Image
Image(filename='mmv.png')


# In[136]:


import pandas as pd
"""
A framework script that tags the data points according to the gear and assigns it a color and plots the data. 
The gear detection is done by assuming the borders generated using any of the algorithms and placed in
the borders array. 
"""

get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt


def get_gear(entry, borders):
    if entry['rpm'] == 0:
        return 0
    rat = entry['speed'] / entry['rpm'] * 1000
    if np.isnan(rat) or np.isinf(rat):
        return 0
    for i in range(0, len(borders)):
        if rat < borders[i] :
            return i + 1
    return 0

num_trips = 10
df = pd.read_csv("C:/Users/Kingsley/Desktop/allcars.csv", index_col=0)
obddata = df[df['tripID']<num_trips]

# borders = get_segment_borders(obddata)
borders = [7.070124715964856, 13.362448319790191, 19.945056624926686, 27.367647318253834, 32.17327586520911]

obddata_wgears = obddata
obddata_wgears['gear'] = obddata.apply(lambda x : get_gear(x, borders), axis=1)

# print(obddata_wgears)

colors = [x * 50 for x in obddata_wgears['gear']]
plt.scatter(obddata_wgears['rpm'], obddata_wgears['speed'], c=colors)
plt.plot()


# In[ ]:




