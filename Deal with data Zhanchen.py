from __future__ import division
import pandas as pd
import numpy as np
from sklearn import svm
import time
import matplotlib.pyplot as plt

CATEGORIES = ['WARRANTS', 'OTHER OFFENSES', 'LARCENY/THEFT', 'VEHICLE THEFT', 'VANDALISM', 'NON-CRIMINAL', 'ROBBERY', 'ASSAULT', 'WEAPON LAWS', 'BURGLARY', 'SUSPICIOUS OCC', 'DRUNKENNESS', 'FORGERY/COUNTERFEITING', 'DRUG/NARCOTIC', 'STOLEN PROPERTY', 'SECONDARY CODES', 'TRESPASS', 'MISSING PERSON', 'FRAUD', 'KIDNAPPING', 'RUNAWAY', 'DRIVING UNDER THE INFLUENCE', 'SEX OFFENSES FORCIBLE', 'PROSTITUTION', 'DISORDERLY CONDUCT', 'ARSON', 'FAMILY OFFENSES', 'LIQUOR LAWS', 'BRIBERY', 'EMBEZZLEMENT', 'SUICIDE', 'LOITERING', 'SEX OFFENSES NON FORCIBLE', 'EXTORTION', 'GAMBLING', 'BAD CHECKS', 'TREA', 'RECOVERED VEHICLE', 'PORNOGRAPHY/OBSCENE MAT']
HORI1 = 37.8086250596 # Use max(data['Y']) you get extract this number
HORI4 = 37.7078790224
VERTI4 = -122.370355487
VERTI1 = -122.513642064
ROWS = 3
COLS = 3
HORI_DIS = (HORI1 - HORI4) / ROWS
VERTI_DIS = (VERTI4 - VERTI1) / COLS

# The HORI1 is the first horizontal line from top to down
# The VERTI1 is the first vertical line from left to right

#Use to count the length of the data. Ignore this function
def check_num():
	counts = [0] * 13
	def years(year, word, n):
		if year in word:
			counts[n] += 1
	with open("test.csv") as f:
		for i in f:
			words = i.split(",")
			for i in xrange(7):
				years("200" + str(i + 3), words[1], i)
			for i in xrange(6):
				years("201" + str(i), words[1], i + 7)
	print counts
	print sum(counts)

#use to create a pickle file that contained cleaned data
def clean_data():
	data = pd.read_csv("train.csv")
	print len(data)
	data = data[data['X'] < -122.368] # Kick out the useless data
	data = data[data['Y'] < 37.8113]
	print len(data)
	print data.head()
	data.to_pickle('train_clean.pickle')

# Get the list of the unique categories
def get_categories(data):
	cat = data.drop_duplicates('Category')['Category'].tolist()
	print cat

# Plot all the categories
def all_categories(data):
	for i in xrange(39):
		plt.figure()
		branch = data[data['Category'] == CATEGORIES[i]]
		branch.plot.scatter(x='X', y='Y')
		plt.title(CATEGORIES[i])

# Split the regions into 9 pieces. Return the list of the regions.
# Each region is a dataframe
def catergorize(data):
	ax = plt.figure().add_subplot(111)
	#create horizontal lines to separate regions
	hori_lines = [HORI1]
	for i in xrange(ROWS):
		hori_lines.append(hori_lines[i] - HORI_DIS)
	rows = [(hori_lines[i], hori_lines[i + 1]) for i in xrange(ROWS)] 
	
	#create vertical lines to separate regions
	verti_lines = [VERTI1]
	for i in xrange(COLS):
		verti_lines.append(verti_lines[i] + VERTI_DIS)
	cols = [(verti_lines[i], verti_lines[i + 1]) for i in xrange(COLS)] 

	region_index = [(i, j) for i in rows for j in cols]
	regions = [data[(data['Y'] < i[0]) & (data['Y'] > i[1]) & (data['X'] > j[0]) & (data['X'] < j[1])] for i, j in region_index]
	colors = ['blue', 'red', 'white', 'green', 'grey', 'pink', 'purple','black','brown']
	print_regions(regions, colors, ax)
	return regions

#Convert the Dates in the data from string to time. However this code is not mature
def convertDate(regions):
	for region in regions:
		region['Dates'] = map(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')), region['Dates'])

# To print out the region with different colors 
def print_regions(regions, colors, ax):
	for i, color in zip(regions, colors):
		i.plot.scatter(ax=ax, x='X', y='Y', color=color)
	plt.show()
		
# Runaway is the one I filtered from the main data so the testing will run faster
# You can do it directly with main data
data = pd.read_pickle("runaway.pickle")
catergorize(data)



