import pandas as pd

def monthwise_species_checklists(filtered_dataframe, months, unique_species_list):
    """ 
    
    Parameters
    ----------
    filtered_dataframe          :dataframe
                                dataframe which only has following column 'Month_Num','MONTH','SCIENTIFIC NAME','SAMPLING EVENT IDENTIFIER'
               
    month                       :str
                                month of the year
               
    unique_species_list         :list
                                list of all species in ABA(code 1 and code 2) alphabetically sorted

                
               
    Returns
    -------
    
    df_with_species_checkist_col :dataframe
                                 dataframe with additional column for complete checklist per species per month

    """
    dict_species = {}
    for i, month in enumerate(months):      
        for j, species in enumerate(unique_species_list):
            checklists = filtered_dataframe[
                (filtered_dataframe['Month_Num'] == i+1) &
                (filtered_dataframe['MONTH'] == month) &
                (filtered_dataframe['SCIENTIFIC NAME'] == species) 
            ]
            
            species_c_check = checklists['SAMPLING EVENT IDENTIFIER'].nunique()
            dict_species[species, month] = species_c_check
        
        # Monthwise list of all unique species from ABA code 1 and 2
        filtered_dataframe_unique_species = filtered_dataframe.loc[(filtered_dataframe['MONTH'] == month)].drop_duplicates(subset=['SCIENTIFIC NAME'])

        # Add species complete checklists column row by row
        if 'SPECIES_COMPLETE_CHECKLISTS' not in filtered_dataframe_unique_species.columns:
            filtered_dataframe_unique_species['SPECIES_COMPLETE_CHECKLISTS'] = filtered_dataframe_unique_species.set_index(['SCIENTIFIC NAME','MONTH']).index.map(dict_species.get)

        if month =='January':
            df_with_species_checkist_col = filtered_dataframe_unique_species
        else:
            df_with_species_checkist_col = pd.concat([df_with_species_checkist_col, filtered_dataframe_unique_species], ignore_index=True)

    return df_with_species_checkist_col