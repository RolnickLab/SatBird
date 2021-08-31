"""
Code which creates hotspot pickle objects for species information and complete cheklists
"""

# TO DO: Check why positional argument and keyword argument cant be fixed
import fnmatch
import os
import pandas as pd
from data_processing.utils.get_each_hotspot import each_hotspot
from data_processing.utils.get_dataframe_as_pkl import create_dataframes_as_pkl
from data_processing.utils.helper_data import get_list_of_hotspot


def main():
    # List of all 'species' (including the ones which had incorrect names corrected)
    all_species = pd.read_csv('data_processing/utils/684_species_with_correct_names.csv')
    all_species_list = all_species['scientific_name'].tolist()

    hotspot_path = '/miniscratch/srishtiy/hotspot_csv_data/'

    list_of_hotspots = get_list_of_hotspot(hotspot_path)
    print("Number of hotspots:", len(list_of_hotspots))
    
    for every_hotspot in list_of_hotspots:
        hotspot_id = every_hotspot
        categoryname = "species"
        
        df_species_pkl_path                 = "/miniscratch/srishtiy/species_object/" + hotspot_id + "_original_hotspot_data.pkl"
        complete_dataframe_sorted_pkl_path  = "/miniscratch/srishtiy/species_object/" + hotspot_id + "_all_aba_hotspot_data.pkl"
        df_complete_checklists_pkl_path     = "/miniscratch/srishtiy/species_object/" + hotspot_id + "_complete_checklists.pkl"
        each_hotspot_data_path_pkl_path     = "/miniscratch/srishtiy/species_object/" + hotspot_id + "_hotspot_object.pkl"

        df_species, complete_dataframe_sorted, dataframe_with_complete_checklists = each_hotspot(hotspot_path,hotspot_id,all_species_list,categoryname)

        create_dataframes_as_pkl(hotspot_id, 
                                 df_species, 
                                 df_species_pkl_path,
                                 complete_dataframe_sorted, 
                                 complete_dataframe_sorted_pkl_path,
                                 dataframe_with_complete_checklists,
                                 df_complete_checklists_pkl_path,
                                 each_hotspot_data_path_pkl_path)
        
        print('Hotspot id done:', hotspot_id)


if __name__ == "__main__":
    main()