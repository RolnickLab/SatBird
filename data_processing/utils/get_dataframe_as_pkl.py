import pandas as pd
from  data_processing.utils.get_saved_pickle import save_pickle

def create_dataframes_as_pkl(hotspot_id, 
                             df_species, 
                             df_species_path,
                             complete_dataframe_sorted, 
                             complete_dataframe_sorted_path,
                             df_with_complete_checklists, 
                             df_complete_checklists_path, 
                             df_hotspot_data_path):
    """Save the hotspot informations as pickles.
    - Save the original dataframe
    - Save the merged data frame with each month having 684 species
    - Save the checklists information (it has species wise and monthwise)
    
    Parameters
    ----------
    hotspot_id                     :str
                                   locality ID of the hotspot

    df_species                     :dataframe
                                   Original hotspot information
                                   
    df_species_path                :str
                                   Path of dataframe with original hotspot information
                                   
    
    complete_dataframe_sorted      :dataframe
                                   dataframe with hotspot information including absent species (i.e. includin ABA speies not present in original data)
    
    complete_dataframe_sorted_path :str
                                   path of dataframe with hotspot information including absent species (i.e. includin ABA speies not present in original data)
                                   
    df_with_complete_checklists    :dataframe
                                   dataframe with complete checklist information
                                   
    df_complete_checklists_path    :str
                                   path of dataframe with complete checklist information
                                    
    df_hotspot_data_path           :str
                                   path of pickle that saves all the hotspot info mentioned above
                 
    """


    # Create a list of hotspot information
    data = {'Hotspot_name'                 : df_species['LOCALITY'].unique(),
            'Hotspot_id'                   : df_species['LOCALITY ID'].unique(),
            'Latitude'                     : df_species['LATITUDE'].unique(),
            'Longitude'                    : df_species['LONGITUDE'].unique(),
            'State'                        : df_species['STATE'].unique(),
            'Country'                      : df_species['COUNTRY'].unique(),
            'Complete_checklists_monthwise': df_complete_checklists_path,
            'Original_hotspot_data'        : df_species_path,
            'Hotspot_data_with_all_species': complete_dataframe_sorted_path
           }

    df_hotspot_data = pd.DataFrame(data)
    
    save_pickle(df_species, df_species_path)
    save_pickle(complete_dataframe_sorted,complete_dataframe_sorted_path)
    save_pickle(df_with_complete_checklists,df_complete_checklists_path)
    save_pickle(df_hotspot_data,df_hotspot_data_path)
    
    print("processing complete..")