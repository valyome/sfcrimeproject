import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss
import numpy as np
from pybrain.datasets import ClassificationDataSet

CATEGORIES = [  'WARRANTS',                     'OTHER OFFENSES',               'LARCENY/THEFT',            'VEHICLE THEFT',        # 0 -  3
                'VANDALISM',                    'NON-CRIMINAL',                 'ROBBERY',                  'ASSAULT',              # 4 -  7
                'WEAPON LAWS',                  'BURGLARY',                     'SUSPICIOUS OCC',           'DRUNKENNESS',          # 8 - 11
                'FORGERY/COUNTERFEITING',       'DRUG/NARCOTIC',                'STOLEN PROPERTY',          'SECONDARY CODES',      #12 - 15
                'TRESPASS',                     'MISSING PERSON',               'FRAUD',                    'KIDNAPPING',           #16 - 19
                'RUNAWAY',                      'DRIVING UNDER THE INFLUENCE',  'SEX OFFENSES FORCIBLE',    'PROSTITUTION',         #20 - 23
                'DISORDERLY CONDUCT',           'ARSON',                        'FAMILY OFFENSES',          'LIQUOR LAWS',          #24 - 27
                'BRIBERY',                      'EMBEZZLEMENT',                 'SUICIDE',                  'LOITERING',            #28 - 31
                'SEX OFFENSES NON FORCIBLE',    'EXTORTION',                    'GAMBLING',                 'BAD CHECKS',           #32 - 35
                'TREA',                         'RECOVERED VEHICLE',            'PORNOGRAPHY/OBSCENE MAT']                          #36 - 38

#Load Data with pandas, and parse the first column into datetime

train_csv = pd.read_csv('data/train.csv', parse_dates = ['Dates'],low_memory=False)
test_csv = pd.read_csv('data/test.csv', parse_dates = ['Dates'],low_memory=False)

def normalize(data_list):
	result = (data_list - data_list.mean()) / (data_list.max() - data_list.min())
	return result

# restructures data
le = preprocessing.LabelEncoder()

crime = le.fit_transform(train_csv.Category) # numbered crime categories
day = pd.DataFrame(le.fit_transform(train_csv.DayOfWeek))
district = pd.DataFrame(le.fit_transform(train_csv.PdDistrict))
hour = train_csv.Dates.dt.hour
location_x = normalize(train_csv["X"])
location_y = normalize(train_csv["Y"])

features = pd.concat((hour, day, district, location_x, location_y), axis=1)
features.columns = ['Hour', 'Day', 'PdDistrict', 'X', 'Y']	# renaming the columns

classify = ClassificationDataSet(5, class_labels = CATEGORIES)

for i in range(len(crime)):
        classify.appendLinked([hour[i], day[i], district[i], location_x[i], location_y[i]], [crime[i]])


validation, train = classify.splitWithProportion(0.25)
validation._convertToOneOfMany()
train._convertToOneOfMany()

print classify

# naivebayesianmodel = naive_bayes.BernoulliNB()
# naivebayesianmodel.fit(training[features], training['crime'])
# predicted = np.array(naivebayesianmodel.predict_proba(validation[features]))

print log_loss(validation['crime'], predicted)

