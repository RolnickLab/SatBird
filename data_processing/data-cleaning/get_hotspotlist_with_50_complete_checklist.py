import os
import sys
import datetime
import time
import pandas as pd
from collections import OrderedDict


# Read complete checklists file
df = pd.read_csv('./complete-checklists_Jan2021.txt', delimiter = "\t", keep_default_na=False)

# n_checklists_Apr2021 is the number of complete checklist per hotspot
# n_checklists_Apr2021 is same for both Jan and Apr -- tested
# n_checklists_Apr2021 is from R code
df_nc_hotspot = pd.read_csv('./n_checklists_Apr2021.csv', keep_default_na=False)

# Sort by number of complete checklists
df_nc_hotspot_sorted = df_nc_hotspot.sort_values(by=['n'], ascending = False)

# Rename header of the dataframe
df_new = df_nc_hotspot_sorted.rename({'locality_id': 'LOCALITY ID', 'locality': 'LOCALITY'}, axis=1)

# Merge two dataframes such that each hotspot has:
# - 'number' of complete checklists
# -  other informations like latitude longitude information etc

# This step takes time to execute. May lead to kernel restarting
df_checklist_count = df_new.merge(df, on ='LOCALITY', how='inner') 

# Select only US based hotspots
df_checklist_usa = df_checklist_count.loc[df_checklist_count['country'] == 'United States']

# Drop duplicate columns when data was merged
df_checklist_usa_uc = df_checklist_usa.drop('LOCALITY ID_y', 1) # uc in df_checklist_usa_uc: unique column

### Drop duplicate rows
df_checklist_usa_unique = df_checklist_usa.drop_duplicates('LOCALITY ID_x')
df_checklist_usa_unique = df_checklist_usa_unique.sort_values(by=['n'], ascending = False)

# Selectonly continental USA
# c_usa = continental USA
df_c_usa = df_checklist_usa_unique[(df_checklist_usa_unique.STATE != "Alaska") &
                                   (df_checklist_usa_unique.STATE != "District of Columbia") &
                                   (df_checklist_usa_unique.STATE != "Hawaii")]

# All hotspots with their number of checklists in Continental USA
df_usa_count = df_c_usa.sort_values(by=['n'], ascending = False)
print("Number of hotspots in continental USA", len(df_usa_count))

# Places with complete checklist > 50
threshold = 49
df_usa_count_threshold = df_usa_count[(df_usa_count['n'] > threshold)]

# Make a list of hostspot IDs
list_loc = df_usa_count_threshold['LOCALITY ID_x'].to_list()
print ("Length of list of locality:", len(list_loc))

my_df = pd.DataFrame(list_loc)
my_df.to_csv('hotspotlist_with_50_complete_checklists.csv', index=False, header=['LOCALITY_ID'])