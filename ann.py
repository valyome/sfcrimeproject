import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
import numpy as np
import pickle
import sys
import os

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

def normalize(data_list):
        result = (data_list - data_list.mean()) / (data_list.max() - data_list.min())
        return result

######################################################################

# loads data and parses the first column into datetime
print "Reading data."
train_csv = pd.read_csv('data/train.csv', parse_dates = ['Dates'],low_memory=False)
test_csv = pd.read_csv('data/test.csv', parse_dates = ['Dates'],low_memory=False)

# restructures data
print "Restructuring data."
le_crime = preprocessing.LabelEncoder()
le_day = preprocessing.LabelEncoder()
le_district = preprocessing.LabelEncoder()


crime = pd.DataFrame(le_crime.fit_transform(train_csv.Category)) # numbered crime categories
day = pd.DataFrame(le_day.fit_transform(train_csv.DayOfWeek))
district = pd.DataFrame(le_district.fit_transform(train_csv.PdDistrict))
hour = train_csv.Dates.dt.hour
location_x = normalize(train_csv["X"])
location_y = normalize(train_csv["Y"])

features = pd.concat((hour, day, district, location_x, location_y), axis=1)
features.columns = ['Hour', 'Day', 'PdDistrict', 'X', 'Y']	# renaming the columns

# builds classification dataset
training = pd.concat((features, crime), axis=1)
training.columns = ['Hour', 'Day', 'PdDistrict', 'X', 'Y', 'Crime']

print "Splitting dataset."
train, validation = train_test_split(training, train_size=.80, random_state=0)

train_features = train.ix[:, 0:5]
train_crime = train.ix[:, 5]

train_features_small = train_features.iloc[:10000]
train_crime_small = train_crime.iloc[:10000]

validation_features = validation.ix[:, 0:5]
validation_crime = validation.ix[:, 5]

validation_features_small = validation_features.iloc[:2000]
validation_crime_small = validation_crime.iloc[:2000]



TRAIN_AND_DUMP = False # else read from dump
classifier_dump_file_location = "classifier_4.pkl"

if TRAIN_AND_DUMP:
        print "Training classifier."
        classifier = MLPClassifier(activation='logistic', hidden_layer_sizes=(22,22,22,22,))
        classifier.fit(train_features, train_crime)

        print "Dumping to %s." % classifier_dump_file_location
        classifier_dump_file = open(classifier_dump_file_location, "wb")
        pickle.dump(classifier, classifier_dump_file)
        classifier_dump_file.close()
        ### !!!!!! ###
        sys.exit()
        ### !!!!!! ###
else:
        print "Reading classifier."
        classifier_dump_file = open(classifier_dump_file_location, "rb")
        classifier = pickle.load(classifier_dump_file)
        classifier_dump_file.close()

print "Predicting."
predicted_proba = np.array(classifier.predict_proba(validation_features))
predicted_categories = np.array(classifier.predict(validation_features))

print accuracy_score(validation_crime, predicted_categories)

predicted_proba_most_probable_categories = []
for element in predicted_proba:
        element_processed = []
        for i in range(len(element)):
                element_processed.append([element[i], i]) # pairs up category probabilities with category indices
        element_processed.sort()
        most_probable_categories_count = 15
        predicted_proba_most_probable_categories.append(element_processed[:-most_probable_categories_count-1:-1]) # most probable n categories

# relaxed accuracy test for if the most probable categories include the true category
accuracy_count = 0
for i in range(len(validation_crime)):
        predicted_category_list = [element[1] for element in predicted_proba_most_probable_categories[i]]
        if validation_crime.iloc[i] in predicted_category_list:
                accuracy_count += 1

        os.system("clear")
        print "############################################################"
        print validation_features.iloc[i]
        print "---------------------------"
        print "ACTUAL CRIME:    %s" % (le_crime.inverse_transform([validation_crime.iloc[i]])[0])
        print "PREDICTED CRIME: %s" % (le_crime.inverse_transform(predicted_category_list))
        raw_input()


accuracy = accuracy_count/float(len(validation_crime))
print accuracy


# print "Converting validation set class labels."
# validation._convertToOneOfMany()

# print "Converting training set class labels."
# train._convertToOneOfMany()


# naivebayesianmodel = naive_bayes.BernoulliNB()
# naivebayesianmodel.fit(training[features], training['crime'])
# predicted = np.array(naivebayesianmodel.predict_proba(validation[features]))

