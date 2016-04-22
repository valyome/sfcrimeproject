
# coding: utf-8

# In[1]:

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
import numpy as np

#Load Data with pandas, and parse the first column into datetime


# In[2]:

train=pd.read_csv('train.csv', parse_dates = ['Dates'],low_memory=False)
test=pd.read_csv('test.csv', parse_dates = ['Dates'],low_memory=False)


# # Implementing Decision Tree Model

# In[4]:

from sklearn import metrics
#Convert labels to numbers
le = preprocessing.LabelEncoder()
#convert crime, day, district to numbers
crime = le.fit_transform(train.Category)
day = pd.DataFrame(le.fit_transform(train.DayOfWeek))
district = pd.DataFrame(le.fit_transform(train.PdDistrict))
year = train.Dates.dt.year
hour = train.Dates.dt.hour
#create training data by joining hour, year, day, and district
train_data = pd.concat([hour, year, day, district], axis=1)
train_data['crime']=crime
train_data.columns = ['hour','year','day', 'PdDistrict','crime']
features = ['hour','year','PdDistrict']
from sklearn import tree
#splitting training set into training and validation 
training1, validation1 = train_test_split(train_data, train_size=.80,random_state=0)
#creating decision tree model
decisiontree = tree.DecisionTreeClassifier(max_depth = 7)
decisiontree.fit(training1[features], training1['crime'])
predictedtreeprob = np.array(decisiontree.predict_proba(validation1[features]))
predictedtree = np.array(decisiontree.predict(validation1[features]))
print metrics.log_loss(validation1['crime'], predictedtreeprob)
print metrics.accuracy_score(validation1['crime'], predictedtree)


# In[8]:

#performing cross validation: calculate cross validation score
from sklearn.cross_validation import cross_val_score
decisiontreescores = cross_val_score(decisiontree, training1[features], training1['crime'])
decisiontreescores


# # Implementing Naive Bayesian Model

# In[10]:

#creating dummy variables
dummydays = pd.get_dummies((train.DayOfWeek))
dummydistrict = pd.get_dummies(list(train.PdDistrict))
dummyhour = pd.get_dummies(list(train.Dates.dt.hour))
#pd.set_option('display.max_columns', 50)
naivebayesiantrain_data = pd.concat([dummyhour, dummydays, dummydistrict], axis=1)
naivebayesiantrain_data['crime']=crime


# In[ ]:

originaldatanaivebayes = naive_bayes.BernoulliNB()
originaldatanaivebayes.fit(training1[features], training1['crime'])
naivebayespredictedprob = np.array(originaldatanaivebayes.predict_proba(validation1[features]))
naivebayespredicted = np.array(originaldatanaivebayes.predict(validation1[features]))
print metrics.log_loss(validation1['crime'], naivebayespredictedprob)
print metrics.accuracy_score(validation1['crime'], naivebayespredicted)
#performing cross validation: calculate cross validation score
naivebayesscores = cross_val_score(originaldatanaivebayes, training1[features], training1['crime'])
naivebayesscores


# In[ ]:

training, validation = train_test_split(naivebayesiantrain_data, train_size=.60,random_state=0)
naivebayesianfeatures = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
 'Wednesday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION',
 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']

naivebayesianmodel = naive_bayes.BernoulliNB()
naivebayesianmodel.fit(training[naivebayesianfeatures], training['crime'])
predictedprob = np.array(naivebayesianmodel.predict_proba(validation[naivebayesianfeatures]))
predicted = np.array(naivebayesianmodel.predict(validation[naivebayesianfeatures]))

print metrics.log_loss(validation['crime'], predictedprob)
print metrics.accuracy_score(validation['crime'], predicted)


# In[ ]:

from sklearn.cross_validation import cross_val_score
naivebayesianscores = cross_val_score(naivebayesianmodel, training[naivebayesianfeatures], training['crime'])
naivebayesianscores


# # Implement Random Forest
# 

# In[ ]:

from sklearn import ensemble
Randomforest = ensemble.RandomForestClassifier(n_estimators=100,min_samples_split=1)
Randomforest.fit(training1[features], training1['crime'])
predictedtreeprob = np.array(Randomforest.predict_proba(validation1[features]))
predictedtree = np.array(Randomforest.predict(validation1[features]))
print "The number of trees in the forest is: ", i
print metrics.log_loss(validation1['crime'], predictedtreeprob),
print metrics.accuracy_score(validation1['crime'], predictedtree)


# In[ ]:

from sklearn.cross_validation import cross_val_score

scores = cross_val_score(Randomforest, training1[features], training1['crime'])

