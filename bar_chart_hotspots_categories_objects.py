#!/usr/bin/env python
# coding: utf-8

"""
This script gets observation count for a particular hostpot (id mentioned), creates bar chart for species oservation permonth between 2000- 2020.
Hotspot information can be accessed via respective classes.
"""

import fnmatch, os, shutil
import pandas as pd
import each_hotspot_df
import matplotlib.pyplot as plt
import seaborn as sns
from data_processing.utils import create_folder
from each_hotspot_dataframe import each_hotspot_df

# List of all 'species' (including the ones which had incorrect names corrected)
all_species = pd.read_csv('data_processing/utils/684_species_with_correct_names.csv')

all_species_list = all_species['scientific_name'].tolist()
# print("length of species:", len(all_species_list))

# Location of indivusla hotspot csv
hotspot_path = '/miniscratch/srishtiy/hotspot_csv_data/'

# Get hotspot id and name
idx = 0
hotspot = os.listdir(hotspot_path)[idx].split('.')[0]

# Use this id to get the respective df
df, df_species_aba = each_hotspot_df(idx, 'species')

# Reorder the dataframe columns
df_species = df_species_aba[['Unnamed: 0', 'LAST EDITED DATE', 'CATEGORY','TAXONOMIC ORDER', 'COMMON NAME', 'SCIENTIFIC NAME', 
                           'SUBSPECIES COMMON NAME', 'SUBSPECIES SCIENTIFIC NAME', 'OBSERVATION COUNT', 'BREEDING BIRD ATLAS CODE', 'BREEDING BIRD ATLAS CATEGORY', 
                           'AGE/SEX', 'COUNTRY', 'COUNTRY CODE', 'STATE', 'STATE CODE', 'COUNTY', 'COUNTY CODE', 'IBA CODE', 'BCR CODE', 'USFWS CODE', 'ATLAS BLOCK', 
                           'LOCALITY', 'LOCALITY ID', 'LOCALITY TYPE', 'LATITUDE', 'LONGITUDE', 'OBSERVATION DATE', 'TIME OBSERVATIONS STARTED', 'OBSERVER ID', 
                           'SAMPLING EVENT IDENTIFIER', 'PROTOCOL TYPE', 'PROTOCOL CODE', 'PROJECT CODE', 'DURATION MINUTES', 'EFFORT DISTANCE KM', 'EFFORT AREA HA', 
                           'NUMBER OBSERVERS','ALL SPECIES REPORTED', 'GROUP IDENTIFIER', 'HAS MEDIA', 'APPROVED', 'REVIEWED', 'REASON', 'TRIP COMMENTS', 
                           'SPECIES COMMENTS','GLOBAL UNIQUE IDENTIFIER']]


# Extract months from the date and Time (to be used in bar chart)
df_species['LAST EDITED DATE'] = pd.to_datetime(df_species['LAST EDITED DATE'])
df_species['Month'] = df_species['LAST EDITED DATE'].dt.month_name()

# Month column doesn't exist by default so create it because we need it cretae bar chart
if 'MONTH' not in df_species:
    df_species.insert(df_species.columns.get_loc('LAST EDITED DATE'), 'MONTH', df_species['Month'] )

state_bird_month = df_species[["LOCALITY ID", "OBSERVATION COUNT", "SCIENTIFIC NAME", "MONTH", "ALL SPECIES REPORTED","LATITUDE", "LONGITUDE"]]
state_bird_month["ALL SPECIES REPORTED"] = pd.to_numeric(state_bird_month["ALL SPECIES REPORTED"])
# state_bird_month = state_bird_month[state_bird_month["ALL SPECIES REPORTED"] == 1]
print(state_bird_month.shape)


# Add all the species to the list (from unique species list)

unique_species_csv =  pd.read_csv('data_processing/utils/unique_species.csv')
unique_species_list = list(unique_species_csv.unique_species)
unique_species_list = all_species_list
print("Number of unique species:", len(unique_species_list))

months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


# For each column if value is not in list, add row
species_not_exist_list = []
for column in state_bird_month['SCIENTIFIC NAME']:
    column = [column]
    if column not in unique_species_list:
        species_not_exist_list.append(column)

print("Number of species to be added:", len(species_not_exist_list))
species_not_exist_list[0]


## Create a new dataframe where existent species not present in certain months, can be given some values
#-------------------------------------------------------------------------------------------------------
def absent_month(all_months, this_month):
    this_month = [this_month]
    abs_month = list(set(all_months) - set(this_month))
    return abs_month
    

# Make DF for species that are in ebird and in ABA code but don't have all months. If months doesn't exists, add month and add 0 the species count
# We need this to have all months for the data

# create list
species_exist_no_month_current_data = []

for idx, seriesdata in state_bird_month.iterrows():
    species = seriesdata['SCIENTIFIC NAME']
    month = seriesdata['MONTH']
    lat   = seriesdata["LATITUDE"]
    long  = seriesdata["LONGITUDE"]
    abs_month_list = []
    abs_month_list = absent_month(months, month)
    for i in range(len(abs_month_list)):
        species_exist_no_month_data = [hotspot, 0, species, abs_month_list[i], 0, lat, long] #dataframe format
        species_exist_no_month_current_data.append(species_exist_no_month_data)

# convert list to dataframe
species_exist_no_month_current_df= pd.DataFrame(species_exist_no_month_current_data, 
                                                columns =["LOCALITY ID", "OBSERVATION COUNT",  "SCIENTIFIC NAME", "MONTH", "ALL SPECIES REPORTED", "LATITUDE", "LONGITUDE"])
species_exist_no_month_current_df.shape
# print(species_exist_no_month_current_df.nunique())
# print(type(species_exist_no_month_current_df['MONTH']))
# print(type(species_exist_no_month_current_df['LATITUDE']))


# concatenate both dataframes to get all species information for all months
frames_first = [state_bird_month, species_exist_no_month_current_df]
concat_species_first = pd.concat(frames_first)


first_concate = 'Yes'

if first_concate == 'Yes':
    state_bird_month_concat = concat_species_first
    state_bird_month_concat.shape
    print(state_bird_month_concat.nunique())
else:
    state_bird_month_concat = state_bird_month
    state_bird_month_concat.shape
    print(state_bird_month_concat.nunique())


# Renaming the datfarme for readabilty ahead
state_bird_month = state_bird_month_concat
state_bird_month.shape


# Sort the filtered data with respect to column
state_bird_month['MONTH'] = pd.Categorical(state_bird_month['MONTH'], categories=months, ordered=True)
state_bird_month.sort_values(by='MONTH',inplace=True) # Sort from Jan - December
state_bird_month['MONTH'] = state_bird_month['MONTH'].astype(object)
# print(state_bird_month.nunique())

# Number the months 
months_dict = {"January":1, "February":2, "March":3, "April":4, "May":5, "June":6,
          "July":7, "August":8, "September":9, "October":10, "November":11, "December":12
         }
state_bird_month_num = state_bird_month.MONTH.map(months_dict)

if 'Month_Num' not in state_bird_month:
    state_bird_month.insert(0, "Month_Num", state_bird_month_num)


# Create bar chart data
state_bird_month_sum = state_bird_month.groupby(['Month_Num','MONTH','SCIENTIFIC NAME', 'LATITUDE', 'LONGITUDE'])[['ALL SPECIES REPORTED']].agg('sum')
print("Shape of bar chart data:", state_bird_month_sum.shape)
species_observation_count = state_bird_month_sum.reset_index()

# Get the matrix of hotspot with species and selected column
matrix = species_observation_count.pivot_table("ALL SPECIES REPORTED", "SCIENTIFIC NAME", "Month_Num")
print(type(matrix))


## Plot bar Chart
#----------------

# Folder where all bar charts will be saved
barchart_folder = 'bar_charts'

# create folder to save barchart figures. it removes a folder if it exists and creates new folder everytime
create_barchart_Colder(barchart_folder)

df_temp = species_observation_count.pivot_table("ALL SPECIES REPORTED", "SCIENTIFIC NAME", "Month_Num")

plt.figure(figsize = (45,40))
plt.title("BAR CHART", size=30)
# sns.light_palette("seagreen", as_cmap=True)
ax = sns.heatmap(df_temp, annot = True, cmap = "Greens", annot_kws={"fontsize":15},
                                                         cbar_kws={"label": 'COMPLETE CHECKLISTS FOR SPECIES'})

# Source: https://stackoverflow.com/questions/48586738/seaborn-heatmap-colorbar-label-font-size
cbar_axes = ax.figure.axes[-1]
ax.figure.axes[-1].yaxis.label.set_size(30)

ax.set_xticklabels(months, rotation='horizontal', fontsize=18)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 15)

plt.savefig("bar_charts/bar_chart_" + hotspot + "_v1.png")


# Define class to access the dataframe
class hotspot_data:
    def __init__(self, all_data):
        self.all_data = df_species
        self.species_count = species_observation_count
        
# Get information species and species count
hotspot_info = hotspot_data(all_data)

# Get  information on each species count month wise
hotspot_info.species_observation_count


