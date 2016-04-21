import csv
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from datetime import timedelta, date, datetime
from pygal.style import LightColorizedStyle
import pygal
import re
import numpy as np

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


def generate_occurrence_list(data_list):
    result = []
    counter = Counter(data_list)
    time_list_ordered = sorted(counter.keys())
    for entry in time_list_ordered:
        result.append((entry, counter[entry]))
    return result

# calculates a rolling average of divisible elements, with given interval length
def calc_rolling_average(data_list, interval_length):
    result = []
    for i in range(len(data_list)-interval_length+1):
        interval = data_list[i:i+interval_length-1]
        interval_average = np.mean(interval)
        result.append(interval_average)
    return result

############################################################

crime_type = CATEGORIES[3]

data = pd.read_pickle('data.pkl')
#data = pd.DataFrame(pd.read_csv('allcrime.csv', parse_dates=True, index_col=4))
#data.to_pickle('data.pkl')
data_series = data[(data.Category == crime_type)]
data_series.sort_index(axis=0, ascending=True, inplace=True)

dates = data_series.index[0:]
occurrence_list = generate_occurrence_list(dates)

rolling_average_interval_length = 30
occurrence_list_count_only = [element[1] for element in occurrence_list]    # separates values out for rolling average
rolling_average_values = calc_rolling_average(occurrence_list_count_only, rolling_average_interval_length)

occurrence_list_rolling_average = []

for i in range(len(rolling_average_values)):
    occurrence_list_rolling_average.append([occurrence_list[i][0], rolling_average_values[i]])

datetimeline = pygal.DateTimeLine(
    x_label_rotation=35, truncate_label=-1,
    x_value_formatter=lambda dt: dt.strftime('%d, %b %Y at %I:%M:%S %p'))
datetimeline.add("Series", occurrence_list_rolling_average)
datetimeline.render_to_file('datetime.svg')
print "done"

# for daterr in dates:
#     if cnt[daterr] == 0:
#         dates_clean.append(daterr)
#     cnt[daterr] += 1

# finallist = []

# for day in dates:
#     finallist.append((day, cnt[day]))


# computes a rolling average list from occurrence list

# for i in range(len(finallist)-rolling_avg_interval_length+1):
# 	cnt_rolling_avg_sublist = [x[1] for x in finallist[i:i+rolling_avg_interval_length-1]]
# 	cnt_rolling_avg = np.mean(cnt_rolling_avg_sublist)
# 	result = (finallist[i][0], cnt_rolling_avg)
# 	finallist_rolling_avg.append(result)

# finaldata = pd.DataFrame(finallist)
# finaldata.pivot(index=0, columns=1)
# finaldata.columns = ['Date','Count']
# finaldata['Date'] = pd.to_datetime(finaldata['Date'])

# line_chart = pygal.DateTimeLine(print_labels=True, print_values=True, style=LightColorizedStyle)
# line_chart.human_readable = True
# line_chart.fill = True
# line_chart.dynamic_print_values = True
# line_chart.show_legend=False
# line_chart.title = 'Vehicle Thefts in SF'
# for dater in dates_clean:
#     #print type(dater)
#     line_chart.add(dater, cnt[dater])
# line_chart.render_to_file('dates.svg')

# #viz:
# line_chart = pygal.Pie(print_labels=True, print_values=True, style=LightColorizedStyle)
# line_chart.human_readable = True
# line_chart.fill = True
# line_chart.dynamic_print_values = True
# line_chart.show_legend=False
# line_chart.title = 'Crime in SF'
# for category in categories:
#     line_chart.add(category, cnt[category])
# #line_chart.add('Category', cnt)
# #print len(zipped)
# #line_chart.render()
# #line_chart.render_to_file('xy_chart.svg')
# #line_chart.render_in_browser()
# string = line_chart.render()
# print string
# print "done"
