def get_absent_species(df_species, unique_species_list):
    """ Returns list of species which are in ABA list but not present incurrent hotspot .
    This is needed because because we want same species information for each month fr all hotspot.
    For absent species, we need to assign 0 values for the months they are not present
    
    Parameters
    ----------
    df_species             :str
                           hotspot dataframe of species category since year 2000

    unique_species_list    :list
                           list of all species in ABA with code 1 and code 2
                
               
    Returns
    -------
    species_not_exist_list :list
                            list of all species present in ABA list (code 1 and 2) but not in current hotspot    
    """

    species_not_exist_list = []
    for species in unique_species_list:
        if species not in df_species['SCIENTIFIC NAME'].values:
            species_not_exist_list.append(species)
    
    return species_not_exist_list