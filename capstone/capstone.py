'''
Created on May 28, 2017
capstone.py - Code for Udacity MLND Capstone Project
@author: Shamit Patel
'''

import numpy as np
import pandas as pd
from sklearn import preprocessing

# Read the data
properties = pd.read_csv('properties_2016.csv',index_col='parcelid')
train_data = pd.read_csv('train_2016.csv')

parcelids = train_data['parcelid']
logerrors = np.array(train_data['logerror'])
transactiondates = np.array(train_data['transactiondate'])

df = pd.DataFrame(data=properties,index=parcelids)

# Fill missing values with 0
df = df.fillna(0)

# Convert alphanumeric features to numerical values
df['hashottuborspa'] = pd.to_numeric(df['hashottuborspa'])
df['propertycountylandusecode'] = pd.Categorical(df['propertycountylandusecode'])
df['propertycountylandusecode'] = df['propertycountylandusecode'].cat.codes
df['propertyzoningdesc'] = pd.Categorical(df['propertyzoningdesc'])
df['propertyzoningdesc'] = df['propertyzoningdesc'].cat.codes
df['taxdelinquencyflag'] = pd.Categorical(df['taxdelinquencyflag'])
df['taxdelinquencyflag'] = df['taxdelinquencyflag'].cat.codes

# Remove unneeded features
features = df.drop('parcelid')
features = features[features.columns[0:56]]

# Separate categorical/noncategorical features
categoricalFeatures = ['airconditioningtypeid', 'architecturalstyletypeid', 'buildingqualitytypeid', 'buildingclasstypeid', 'decktypeid',
'fips', 'fireplaceflag', 'garagetotalsqft', 'hashottuborspa', 'heatingorsystemtypeid', 'pooltypeid10', 'pooltypeid2', 'pooltypeid7', 'propertycountylandusecode',
'propertylandusetypeid', 'propertyzoningdesc', 'rawcensustractandblock', 'regionidcounty', 'regionidcity', 'regionidzip', 'regionidneighborhood', 'storytypeid', 
'typeconstructiontypeid', 'unitcnt', 'yardbuildingsqft17', 'yardbuildingsqft26', 'taxdelinquencyflag']
noncategorical_features = features.drop(categoricalFeatures,axis=1)
categorical_features = features[categoricalFeatures]

labels = logerrors

# Scale noncategorical features to unit variance
noncategorical_features_scaled = preprocessing.scale(noncategorical_features)

# Concatenate categorical/scaled noncategorical features together
features = np.concatenate([noncategorical_features_scaled, np.array(categorical_features)],axis=1)

# Split data into training and test sets using October 1 as the split point
october_1 = np.where(transactiondates=='2016-10-01')[0][0]
training_features = features[:october_1]
training_labels = labels[:october_1] 
testing_features = features[october_1:]
testing_labels = labels[october_1:]

# Generate reduced features using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced_training_features = pca.fit_transform(training_features)
reduced_testing_features = pca.fit_transform(testing_features)

# Visualize first 10 principal components of features
import visuals as vs
featuresDF = pd.DataFrame(data=features)
pca = PCA(n_components=10)
pca.fit(featuresDF)
pca_samples = pca.transform(featuresDF)
pca_results = vs.pca_results(featuresDF, pca)
 
# Benchmark Model
from sklearn.metrics import mean_absolute_error
preds = [np.mean(training_labels)] * len(testing_labels)
print 'Benchmark Results:'
print 'MAE:', mean_absolute_error(testing_labels,preds)
 
# Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(training_features, training_labels)
preds = lin_reg.predict(testing_features)
score = lin_reg.score(testing_features,testing_labels)
print 'Linear Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds), '\n'
 
# PCA + Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(reduced_training_features, training_labels)
preds = lin_reg.predict(reduced_testing_features)
score = lin_reg.score(reduced_testing_features,testing_labels)
print 'PCA + Linear Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds)
 
# Ridge Regression
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(training_features, training_labels)
preds = ridge.predict(testing_features)
score = ridge.score(testing_features,testing_labels)
print 'Ridge Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds), '\n'
 
# PCA + Ridge Regression
ridge = Ridge()
ridge.fit(reduced_training_features, training_labels)
preds = ridge.predict(reduced_testing_features)
score = ridge.score(reduced_testing_features,testing_labels)
print 'PCA + Ridge Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds)
 
# Lasso Regression
from sklearn.linear_model import Lasso
lasso = Lasso(max_iter=2500,alpha=2)
lasso.fit(training_features, training_labels)
preds = lasso.predict(testing_features)
score = lasso.score(testing_features,testing_labels)
print 'Lasso Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds), '\n'
 
# PCA + Lasso Regression
lasso = Lasso(max_iter=2500,alpha=2)
lasso.fit(reduced_training_features, training_labels)
preds = lasso.predict(reduced_testing_features)
score = lasso.score(reduced_testing_features,testing_labels)
print 'PCA + Lasso Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds)
 
# Elastic Net
from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet(l1_ratio=0.25)
elasticnet.fit(training_features, training_labels)
preds = elasticnet.predict(testing_features)
score = elasticnet.score(testing_features,testing_labels)
print 'ElasticNet Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds), '\n'
 
# PCA + Elastic Net
elasticnet = ElasticNet(l1_ratio=0.5)
elasticnet.fit(reduced_training_features, training_labels)
preds = elasticnet.predict(reduced_testing_features)
score = elasticnet.score(reduced_testing_features,testing_labels)
print 'PCA + ElasticNet Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds)
 
# Least-Angle Regression (LARS)
from sklearn.linear_model import Lars
lars = Lars()
lars.fit(training_features, training_labels)
preds = lars.predict(testing_features)
score = lars.score(testing_features,testing_labels)
print 'LARS Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds), '\n'
 
# PCA + LARS
lars = Lars()
lars.fit(reduced_training_features, training_labels)
preds = lars.predict(reduced_testing_features)
score = lars.score(reduced_testing_features,testing_labels)
print 'PCA + LARS Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds)
 
# Orthogonal Matching Pursuit
from sklearn.linear_model import OrthogonalMatchingPursuit
omp = OrthogonalMatchingPursuit()
omp.fit(training_features, training_labels)
preds = omp.predict(testing_features)
score = omp.score(testing_features,testing_labels)
print 'Orthogonal Matching Pursuit Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds), '\n'
 
# PCA + Orthogonal Matching Pursuit
omp = OrthogonalMatchingPursuit()
omp.fit(reduced_training_features, training_labels)
preds = omp.predict(reduced_testing_features)
score = omp.score(reduced_testing_features,testing_labels)
print 'PCA + Orthogonal Matching Pursuit Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds)
 
# Bayesian Ridge Regression
from sklearn.linear_model import BayesianRidge
br = BayesianRidge()
br.fit(training_features, training_labels)
preds = br.predict(testing_features)
score = br.score(testing_features,testing_labels)
print 'Bayesian Ridge Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds), '\n'
 
# PCA + Bayesian Ridge Regression
br = BayesianRidge()
br.fit(reduced_training_features, training_labels)
preds = br.predict(reduced_testing_features)
score = br.score(reduced_testing_features,testing_labels)
print 'PCA + Bayesian Ridge Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds)
 
# Stochastic Gradient Descent Regression
from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor()
sgd.fit(training_features, training_labels)
preds = sgd.predict(testing_features)
score = sgd.score(testing_features,testing_labels)
print 'SGD Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds), '\n'
 
# PCA + Stochastic Gradient Descent Regression
sgd = SGDRegressor()
sgd.fit(reduced_training_features, training_labels)
preds = sgd.predict(reduced_testing_features)
score = sgd.score(reduced_testing_features,testing_labels)
print 'PCA + SGD Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds)
 
# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
poly_training_features = poly.fit_transform(training_features)
lr = LinearRegression()
lr.fit(poly_training_features, training_labels)
poly_testing_features = poly.fit_transform(testing_features)
preds = lr.predict(poly_testing_features)
score = lr.score(poly_testing_features,testing_labels)
print 'Polynomial Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds), '\n'
 
# PCA + Polynomial Regression
poly = PolynomialFeatures(degree=2)
poly_training_features = poly.fit_transform(reduced_training_features)
lr = LinearRegression()
lr.fit(poly_training_features, training_labels)
poly_testing_features = poly.fit_transform(reduced_testing_features)
preds = lr.predict(poly_testing_features)
score = lr.score(poly_testing_features,testing_labels)
print 'PCA + Polynomial Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds), '\n'
 
# Support Vector Regression (RBF Kernel) - takes forever!
from sklearn.svm import SVR
svr = SVR()
svr.fit(training_features, training_labels)
preds = svr.predict(testing_features)
score = svr.score(testing_features, testing_labels)
print 'Support Vector Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds)
 
# PCA + Support Vector Regression (RBF Kernel)
svr = SVR()
svr.fit(reduced_training_features, training_labels)
preds = svr.predict(reduced_testing_features)
score = svr.score(reduced_testing_features,testing_labels)
print 'PCA + Support Vector Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds)
 
# KNN Regression
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(training_features, training_labels)
preds = knr.predict(testing_features)
score = knr.score(testing_features,testing_labels)
print 'KNN Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds), '\n'
 
# PCA + KNN Regression
knr = KNeighborsRegressor()
knr.fit(reduced_training_features, training_labels)
preds = knr.predict(reduced_testing_features)
score = knr.score(reduced_testing_features,testing_labels)
print 'PCA + KNN Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds), '\n'

# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(min_samples_split=6)
dtr.fit(training_features, training_labels)
preds = dtr.predict(testing_features)
score = dtr.score(testing_features,testing_labels)
print 'Decision Tree Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds), '\n'
 
# PCA + Decision Tree Regression
dtr = DecisionTreeRegressor(min_samples_split=6)
dtr.fit(reduced_training_features, training_labels)
preds = dtr.predict(reduced_testing_features)
score = dtr.score(reduced_testing_features,testing_labels)
print 'PCA + Decision Tree Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds), '\n'
 
# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(training_features, training_labels)
preds = rfr.predict(testing_features)
score = rfr.score(testing_features,testing_labels)
print 'Random Forest Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds), '\n'
 
# PCA + Random Forest Regression
rfr = RandomForestRegressor()
rfr.fit(reduced_training_features, training_labels)
preds = rfr.predict(reduced_testing_features)
score = rfr.score(reduced_testing_features,testing_labels)
print 'PCA + Random Forest Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds), '\n'
 
# AdaBoost Regression
from sklearn.ensemble import AdaBoostRegressor
abr = AdaBoostRegressor()
abr.fit(training_features, training_labels)
preds = abr.predict(testing_features)
score = abr.score(testing_features,testing_labels)
print 'AdaBoost Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds), '\n'
 
# PCA + AdaBoost Regression
abr = AdaBoostRegressor()
abr.fit(reduced_training_features, training_labels)
preds = abr.predict(reduced_testing_features)
score = abr.score(reduced_testing_features,testing_labels)
print 'PCA + AdaBoost Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds)
 
# Gradient Boosting Regression
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(min_samples_split=6)
gbr.fit(training_features, training_labels)
preds = gbr.predict(testing_features)
score = gbr.score(testing_features,testing_labels)
print 'Gradient Boosting Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds), '\n'
 
# PCA + Gradient Boosting Regression
gbr = GradientBoostingRegressor(min_samples_split=4)
gbr.fit(reduced_training_features, training_labels)
preds = gbr.predict(reduced_testing_features)
score = gbr.score(reduced_testing_features,testing_labels)
print 'PCA + Gradient Boosting Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds)
 
# Multi-Layer Perceptron Regression
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(max_iter=2500)
mlp.n_layers_=75
mlp.fit(training_features, training_labels)
preds = mlp.predict(testing_features)
score = mlp.score(testing_features,testing_labels)
print 'Multi-Layer Perceptron Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds), '\n'
 
# PCA + Multi-Layer Perceptron Regression
mlp = MLPRegressor(max_iter=2500)
mlp.n_layers_=75
mlp.fit(reduced_training_features, training_labels)
preds = mlp.predict(reduced_testing_features)
score = mlp.score(reduced_testing_features,testing_labels)
print 'PCA + Multi-Layer Perceptron Regression Results:'
print 'R2 score:', score
print 'MAE:', mean_absolute_error(testing_labels,preds)