# def each_hotspot_df(idx, category = 'species'):
#     """ Returns a dataframe of category -> species
    
#     Parameters
#     ----------
#     idx              :int
#                      index number of the file in list array
#     category         :str
#                      column name of the datframe being filtered
#                      deafult: "species"
               
#     Returns
#     --------
#     df_species_aba : dataframe
#                      filtered hotspot dataframe with only <category> and 
#                      having scientific name present in all_species.csv
#     """
    
#     os.listdir(hotspot_path)[idx]
#     hotspot_csv = hotspot_path + os.listdir(hotspot_path)[idx]
#     print("csv file: ", hotspot_csv)

#     # Read the hotspot csv
#     df = pd.read_csv(hotspot_csv)
#     print("Original dimention of data:", df.shape)

    
#     # print("DF of original hotspot csv memory in KB: ", df.memory_usage(index=True).sum()/1024)
#     df_species = df.loc[(df['CATEGORY'] == category) & (df['ALL SPECIES REPORTED'] == 1) ]
#     df_species_inlist = df_species.loc[df_species['SCIENTIFIC NAME'].isin(all_species_list)]
#     print("After filtering dimention of data:", df_species_inlist.shape)
    
#     # Only add data till 2000 (and not before that)
#     df_species_aba = df_species_inlist[(df_species_inlist["OBSERVATION DATE"] < "2021-05-01 00:00:00") & 
#                                        (df_species_inlist["OBSERVATION DATE"] > "2000-01-01 00:00:00")]

    
#     return df, df_species_aba