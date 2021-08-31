
def df_create_values_for_absent_species(scientific_name, observation_count, all_species_reported, locality_id, month):
    """ Return the dataframe with filled values for all absent species/months, where observation count and reported species will be 0
    
    Parameters
    ----------
    scientific_name     :str
                        scientific name of the species
    
    observation_count    :int
                         observation count of the species
               
    all_species_reported :int
                         binary value, 1 when all species are reported, 0 when all species are not reported
               
    locality_id          :str
                         locality id of the hotspot
               
    month                :str
                         month of the year
                
               
    Returns
    -------
    df_val_for_absent_species :dataframe
                              dataframe with row values for species/months absent in current hotspot  
    """
    
#     'Unnamed: 0'                    'LAST EDITED DATE',         'CATEGORY',                   'TAXONOMIC ORDER',                'COMMON NAME',      
#     'SCIENTIFIC NAME',              'SUBSPECIES COMMON NAME',   'SUBSPECIES SCIENTIFIC NAME', 'OBSERVATION COUNT',              'BREEDING BIRD ATLAS CODE', 
#     'BREEDING BIRD ATLAS CATEGORY', 'AGE/SEX',                  'COUNTRY',                    'COUNTRY CODE',                   'STATE',                  
#     'STATE CODE',                   'COUNTY',                   'COUNTY CODE',                'IBA CODE',                       'BCR CODE'         ,
#     'USFWS CODE'   ,                'ATLAS BLOCK',              'LOCALITY',                   'LOCALITY ID',                    'LOCALITY TYPE',
#     'LATITUDE',                     'LONGITUDE',                'MONTH',                      'OBSERVATION DATE',               'TIME OBSERVATIONS STARTED', 
#     'OBSERVER ID',                  'SAMPLING EVENT IDENTIFIER','PROTOCOL TYPE',              'PROTOCOL CODE',                  'PROJECT CODE',                 
#     'DURATION MINUTES',             'EFFORT DISTANCE KM',        'EFFORT AREA HA',            'NUMBER OBSERVERS',               'ALL SPECIES REPORTED', 
#     'GROUP IDENTIFIER',             'HAS MEDIA',                 'APPROVED',                  'REVIEWED',                       'REASON',
#     'TRIP COMMENTS',                'SPECIES COMMENTS',          'GLOBAL UNIQUE IDENTIFIER',  'Month'
        
    df_val_for_absent_species = [
                        float('NaN')     ,float('NaN'),        float('NaN'),           float('NaN'),          float('NaN'),
                        scientific_name  ,float('NaN'),        float('NaN'),      observation_count,          float('NaN'),
                        float('NaN')     ,float('NaN'),        float('NaN'),           float('NaN'),          float('NaN'),
                        float('NaN')     ,float('NaN'),        float('NaN'),           float('NaN'),          float('NaN'),
                        float('NaN')     ,float('NaN'),        float('NaN'),           locality_id ,          float('NaN'),
                        float('NaN')     ,float('NaN'),        month,                  float('NaN'),          float('NaN'),
                        float('NaN')     ,float('NaN'),        float('NaN'),           float('NaN'),          float('NaN'),
                        float('NaN')     ,float('NaN'),        float('NaN'),           float('NaN'),          all_species_reported,
                        float('NaN')     ,float('NaN'),        float('NaN'),           float('NaN'),          float('NaN'),
                        float('NaN')     ,float('NaN'),        float('NaN'),            month
                        ]
    return df_val_for_absent_species

