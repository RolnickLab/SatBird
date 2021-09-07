import pandas as pd

def df_with_added_month_column(df_species):
    """ Return the dataframe with an additional column 'MONTH' calculated from datatime value
    
    Parameters
    ----------
    df_species :dataframe
               hotspot dataframe of species category since year 2000
                
               
    Returns
    -------
    df_species :dataframe
               updated hotspot dataframe with added column of 'MONTH'   
    """
    df_species['OBSERVATION DATE'] = pd.to_datetime(df_species['OBSERVATION DATE'])

    # TO DO: Remove last column 'month' because its already is first few columns 
    df_species['Month'] = df_species['OBSERVATION DATE'].dt.month_name()

    # Replace Month with MONTH for consistency
    if 'MONTH' not in df_species:
        df_species.insert(df_species.columns.get_loc('OBSERVATION DATE'), 'MONTH', df_species['Month'] )
    
    return df_species 

