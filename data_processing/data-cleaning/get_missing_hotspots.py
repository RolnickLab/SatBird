import pandas as pd
import os
from os import listdir
from os.path import isfile, join

def find_missing_hotspots(hotspot_csv_files_path):
    """ This file finds the hotspot IDs which are missing once individual CSVs created (for each hotspot data) from ebird data
    -  Note that this calculation is only done for hotspots which are in USA and have number of complete checklists >=50. 
    - `hotspot_path = '/miniscratch/srishtiy/hotspot_csv_data/` = directory where all the individual hotspot csv files created are stored.
    - `hotspots.csv` = file which has all the hotspots found in USA which ave complete checklist >=50


    Parameters
    ----------
    hotspot_csv_files_path     :str
                               path where all individual hotspot csvs are saved 

    Returns
    -------
    diff_hotspot              :list
                              list of all hotspots which have more than 50 complete checklists 
                              but don't have individual CSVs for them in 'hotspot_csv_files_path'.
                              We had few incorrect data reported (e.g. data of Canada marked as USA) or hotspot not matching loations, 
                              so, there will be 60-100 hotspot names in this variable. The correct final list is in 'hotspot_list.csv' 
                              in the code repo
    """
    # List of all such csv files
    hotspot_files = [f for f in listdir(hotspot_csv_files_path) if isfile(join(hotspot_csv_files_path, f))]

    # Remove .csv extention from the list of csv files
    hotspot_names_from_files = [os.path.splitext(each)[0] for each in hotspot_files]
    print(len(hotspot_names_from_files))

    # Hotspotlist_with_50_complete_checklists.csv = all the hotspots in USA with threshold >= 50
    hotspots_more_than_50_checklists_csv = 'hotspotlist_with_50_complete_checklists.csv'
    hotspots_more_than_50_checklists = pd.read_csv(hotspots_more_than_50_checklists_csv , names= ["total_hotspots"])

    # Convert dataframe of all USA hostspots (>=50 complete checklsts) to list
    all_hotspot = hotspots_more_than_50_checklists.total_hotspots.to_list()

    # Find the different in number of hotspots between hotspots present originally and hotspots found after dividing
    # them into individual CSV
    diff_hotspot = list(set(all_hotspot) - set(hotspot_names_from_files))
    
    return diff_hotspot


def main():
    # Path of all individual CSV files for each hotspot
    hotspot_csv_files_path = './hotspot_csv_data/'
    
    diff_hotspot = find_missing_hotspots(hotspot_csv_files_path)
    print("Number of missing hotspots:", len(diff_hotspot))

    # Add these missing hotspots in a csv file 
    missing_hotspot = pd.DataFrame(diff_hotspot, columns=["missing_hotspot"])
    missing_hotspot.to_csv('missing_hotspot.csv', index=False)
    print("Missing hotspot id are saved in missing_hotspot.csv")


if __name__ == "__main__":
    main()