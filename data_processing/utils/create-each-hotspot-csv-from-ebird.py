# This code creates CSV files for each hotspot from the ebird dataset (threshold >= 50)

# Import libraries
from pandas.io.common import EmptyDataError
import pandas as pd
import numpy as np
from pathlib import Path
import os, csv
import shutil
import os.path
import pathlib


# This file has info of only those hotspots where number of checklist is greater than 50
base_folder = '/miniscratch/tengmeli/'
data =   base_folder + 'ecosystem-embedding/data/usa_hotspot_data_final.csv'

# Folder where all csv will be saved
# IMPORTANT: If changing here, change folder name below in main function
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

def main():
    """
    Create individual csv fles from the filtered ebird data
    
    input    : data (csv files)
    chunksize: number of rows processed in each chunk  
    output   : csv files for each hotspot in filtered ebird dataset
    """
    # EBIRD DATA FILE LOCATION
    reader = pd.read_csv(data , 
                         chunksize = 1000
                        )
    
    # Read chunks and save to a new csv
    for i,chunk in enumerate(reader):
        for grp, dfg in chunk.groupby("LOCALITY ID"):
        # if file does not exist write header 
            if not os.path.isfile(f"hotspots_of_usa_hotspot_data_final_csv/{grp}.csv"):
                dfg.to_csv(f"hotspot_all_csv/{grp}.csv", index=False)
            else: # else it exists so append without writing the header
                dfg.to_csv(f"hotspot_all_csv/{grp}.csv",mode='a', index=False, header=False)

                
        # Progress Bar
        if (i% 1000 == 0):
            print("#", end ='')

    print(" Complete reading the file in chunks.")


if __name__=="__main__":
    main()




