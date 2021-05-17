#!/usr/bin/env python
# coding: utf-8

## Get CSV files for each hospot from the adataset where threshold of 50 complete checklist is already set

# Import libraries
from pandas.io.common import EmptyDataError
import pandas as pd
import numpy as np
from pathlib import Path
import os
import shutil
import os.path


# This file has info of only those hotspots where number of checklist is greater than 50
base_folder = '/miniscratch/tengmeli/'
data = base_folder + 'ecosystem-embedding/data/usa_hotspot_data_2.csv'

# Folder where all csv will be saved
csv_folder = 'hotspot_all_csv'

# Delete directory if it exists and create a new one
if os.path.exists(csv_folder) == False:
    print(" Hotspot CSV folder doesn't exist")

if os.path.isdir(csv_folder):
    print("Exists")
    shutil.rmtree(csv_folder)
    print("Deleted")
    
os.mkdir(csv_folder)
print("New dir Created")


# Check that new directory is created
if os.path.exists(csv_folder) == True:
    print(" Hotspot CSV folder exists")

# Create CSV file and save the data in hotspot_all_csv directory
def main():
    # EBIRD DATA FILE LOCATION
    reader = pd.read_csv(data , 
                         chunksize = 1000
                        )
    
    # Read chunks and save to a new csv
    for i,chunk in enumerate(reader):
        for grp, dfg in chunk.groupby("LOCALITY ID"):
        # if file does not exist write header 
            if not os.path.isfile(f"hotspot_all_csv/{grp}.csv"):
                dfg.to_csv(f"hotspot_all_csv/{grp}.csv", index=False)
            else: # else it exists so append without writing the header
                dfg.to_csv(f"hotspot_all_csv/{grp}.csv",mode='a', index=False)

                
        # Progress Bar
        if (i% 1000 == 0):
            print("#", end ='')

    print(" Complete reading the file in chunks..Reading hotspot info now......")
    state_bird_all_info = pd.read_csv(output_path, delimiter= ',')


if __name__=="__main__":
    main()




