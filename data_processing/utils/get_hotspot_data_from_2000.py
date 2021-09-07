import pandas as pd

def each_hotspot_df(hotspot_path, hotspot_id, all_species_list, category):
    """ Returns orifinal dataframe and dataframe of category -> species
    
    Parameters
    ----------
    hotspot_path     :str
                     :path of all hotspot files
                     
                     
    idx              :str
                     :hotspot id number
    
    category         :str
                     column name of the datframe being filtered
                     deafult: "species"
               
    Returns
    -------
    df             : dataframe
                     original dataframe fetched from the csv file    
    
    df_species_aba : dataframe
                     filtered hotspot dataframe with only <category> and 
                     having scientific name (Code 1 and Code 2 from ABA: 684_species_with_correct_names.csv)
    """
    
    hotspot_csv = hotspot_path + hotspot_id +'.csv'

    # Read the hotspot csv
    df = pd.read_csv(hotspot_csv)

    
    # print("DF of original hotspot csv memory in KB: ", df.memory_usage(index=True).sum()/1024)
    df_species = df.loc[(df['CATEGORY'] == category) & (df['ALL SPECIES REPORTED'] == 1) ]
    df_species_inlist = df_species.loc[df_species['SCIENTIFIC NAME'].isin(all_species_list)]
    
    # Only add data till 2000 (and not before that)
    df_species_aba = df_species_inlist[(df_species_inlist["OBSERVATION DATE"] < "2021-05-01 00:00:00") & 
                                       (df_species_inlist["OBSERVATION DATE"] > "2000-01-01 00:00:00")]

    
    return df, df_species_aba