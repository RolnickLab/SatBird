import pandas as pd

from data_processing.utils.helper_data import monthsofyear, hotspot_headers
from data_processing.utils.get_hotspot_data_from_2000 import each_hotspot_df
from data_processing.utils.get_species_not_in_hotspot import get_absent_species
from data_processing.utils.get_month_data_time import df_with_added_month_column 
from data_processing.utils.get_values_absent_species_or_month import df_create_values_for_absent_species
from data_processing.utils.get_monthwise_species_checklist import monthwise_species_checklists


def each_hotspot(hotspot_path, hotspot_id, all_species_list, category):
    """ Compute all the information for each hotspot including complete checklists""
    
    Parameters
    ----------
    hotspot_path                   :str
                                   :path of all hotspot files
                     
                     
    hotspot_id                     :str
                                   :hotspot id/locality id
    
    all_species_list               :str
                                   list of species in ABA checklist code 1 and code 2
                     
    category                       :str
                                   column name of the datframe being filtered
                                   deafult: "species"
               
    Returns
    -------

    df_species                     :dataframe
                                   Original hotspot information
    
    complete_dataframe_sorted      :dataframe
                                   dataframe with hotspot information including absent species (i.e. includin ABA speies not present in original data)

                                   
    df_with_complete_checklists    :dataframe
                                   dataframe with complete checklist information

    """
    
    df, df_species_aba = each_hotspot_df(hotspot_path, hotspot_id, all_species_list, category)

    headers = hotspot_headers()
    months =  monthsofyear()

    # Get rearranged headers for hotspot
    df_species = df_species_aba[headers]

    # Extract Months from the Date and Time
    df_species = df_with_added_month_column(df_species)

    # Make sure ALL SPECIES REPORTED is all numeric data
    df_species["ALL SPECIES REPORTED"] = pd.to_numeric(df_species["ALL SPECIES REPORTED"])
    updated_headers = list(df_species.columns)

    # Sort species list alphabetically
    unique_species_list = sorted(all_species_list)


    # Get absent species information
    species_not_exist_list = get_absent_species(df_species, unique_species_list)

    ### For each month where a species was not observed, add a row in the original dataframe with values 0, where need be
    ## This is necessary so that we have all species observatin values for all months in each hotspot
    # We need to set a flag so that we don't duplicate values
    # E.g. We may have a species which doesn't exist and species which exists but month doesn't
    # Month doesn't exist in both hence if first is true, second shouldn't be computed
    # TO DO: This takes few seconds to run. can this be optimized to make it even faster?

    absent_species_or_months = []
    df_col_for_absent_species = updated_headers


    for month in months:
        for species in unique_species_list:
            flag = True

            # species which doesn't exist
            if (species not in df_species['SCIENTIFIC NAME'].values):
                absent_species_or_months_data = df_create_values_for_absent_species(species,0,0,hotspot_id,month)
                absent_species_or_months.append(absent_species_or_months_data)
                flag = False

            exisiting_month_for_giving_species = df_species.loc[df_species['SCIENTIFIC NAME'] == species, 'MONTH'].unique()

            #if species exists but month doesn't exist
            if month not in exisiting_month_for_giving_species and flag:
                absent_species_or_months_data = df_create_values_for_absent_species(species,0,0,hotspot_id,month)
                absent_species_or_months.append(absent_species_or_months_data)

    df_for_absent_species_or_months = pd.DataFrame(absent_species_or_months, columns = df_col_for_absent_species)

    # Concatenate the original hotspot dataframe and the new dataframe (with information of non existing species or month)
    complete_dataframe_unsorted = pd.concat([df_species, df_for_absent_species_or_months], ignore_index=True)

    # Sort months from Jan - December
    # When you specify the categories, pandas remembers the order of specification as the default sort order.
    complete_dataframe_unsorted['MONTH'] = pd.Categorical(complete_dataframe_unsorted['MONTH'], categories=months, ordered=True)
    complete_dataframe_unsorted.sort_values(by='MONTH',inplace=True) # Sort from Jan - December
    complete_dataframe_sorted = complete_dataframe_unsorted

    ## Map the months (string) with numbers(int) and add as a new col for proper sorting 
    months_dict = {"January":1, "February":2, "March":3, "April":4, "May":5, "June":6,
              "July":7, "August":8, "September":9, "October":10, "November":11, "December":12
             }

    complete_dataframe_month_num = complete_dataframe_sorted.MONTH.map(months_dict)

    if 'Month_Num' not in complete_dataframe_sorted:
        complete_dataframe_sorted.insert(0, "Month_Num", complete_dataframe_month_num)

    ### Calculate Number of Complete Checklist per species monthwise: identified with SAMPLING IDENTIFIER
    filtered_dataframe = complete_dataframe_sorted[['Month_Num','MONTH','SCIENTIFIC NAME','SAMPLING EVENT IDENTIFIER']]

    df_with_species_checkist_col = monthwise_species_checklists(filtered_dataframe, months, unique_species_list)

    ### dict_month_checklists = dictionary of total number of complete checklists per month  
    dict_month_checklists = {}
    for month in months:
        month_df = df_with_species_checkist_col.loc[df_with_species_checkist_col['MONTH'] == month]

        # calculate all unique sampling identifiers for each month
        unique_comp_checklists = month_df['SAMPLING EVENT IDENTIFIER'].dropna().nunique()

        # dic which saves number of complete checklists per month
        dict_month_checklists[month] = unique_comp_checklists


    # Add a column for total complete checklists
    def month(x):
        return dict_month_checklists[x]

    df_with_species_checkist_col["MONTHWISE_COMPLETE_CHECKLISTS"] = df_with_species_checkist_col["MONTH"].apply(month)
    dataframe_with_complete_checklists = df_with_species_checkist_col

    return df_species, complete_dataframe_sorted, dataframe_with_complete_checklists