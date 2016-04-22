import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
import numpy as np

#Load Data with pandas, and parse the first column into datetime

train=pd.read_csv('train.csv', parse_dates = ['Dates'],low_memory=False)
test=pd.read_csv('test.csv', parse_dates = ['Dates'],low_memory=False)


#Convert crime labels to numbers
le = preprocessing.LabelEncoder()
crime = le.fit_transform(train.Category)
day = pd.DataFrame(le.fit_transform(train.DayOfWeek))
district = pd.DataFrame(le.fit_transform(train.PdDistrict))
hour = train.Dates.dt.hour
train_data = pd.concat([hour, day, district], axis=1)
train_data['crime']=crime
train_data.columns = ['hour','day', 'PdDistrict','crime']
train_data[:5]


training, validation = train_test_split(train_data, train_size=.60,random_state=0)
print type(training), type(validation)
features = ['hour','day','PdDistrict']

naivebayesianmodel = naive_bayes.BernoulliNB()
naivebayesianmodel.fit(training[features], training['crime'])
predicted = np.array(naivebayesianmodel.predict_proba(validation[features]))

print log_loss(validation['crime'], predicted)


from sklearn import tree
training1, validation1 = train_test_split(train_data, train_size=.90,random_state=0)
decisiontree = tree.DecisionTreeClassifier(max_depth = 5)
decisiontree.fit(training1[features], training1['crime'])
predictedtree = np.array(decisiontree.predict_proba(validation1[features]))

print log_loss(validation1['crime'], predictedtree)


print training1
print validation1