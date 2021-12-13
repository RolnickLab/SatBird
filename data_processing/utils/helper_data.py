import fnmatch, os
from os import listdir

def hotspot_headers():
    headers =['Unnamed: 0', 'LAST EDITED DATE', 'CATEGORY','TAXONOMIC ORDER', 'COMMON NAME', 'SCIENTIFIC NAME', 
             'SUBSPECIES COMMON NAME', 'SUBSPECIES SCIENTIFIC NAME', 'OBSERVATION COUNT', 'BREEDING BIRD ATLAS CODE', 'BREEDING BIRD ATLAS CATEGORY', 
             'AGE/SEX', 'COUNTRY', 'COUNTRY CODE', 'STATE', 'STATE CODE', 'COUNTY', 'COUNTY CODE', 'IBA CODE', 'BCR CODE', 'USFWS CODE', 'ATLAS BLOCK', 
             'LOCALITY', 'LOCALITY ID', 'LOCALITY TYPE', 'LATITUDE', 'LONGITUDE', 'OBSERVATION DATE', 'TIME OBSERVATIONS STARTED', 'OBSERVER ID', 
             'SAMPLING EVENT IDENTIFIER', 'PROTOCOL TYPE', 'PROTOCOL CODE', 'PROJECT CODE', 'DURATION MINUTES', 'EFFORT DISTANCE KM', 'EFFORT AREA HA', 
             'NUMBER OBSERVERS','ALL SPECIES REPORTED', 'GROUP IDENTIFIER', 'HAS MEDIA', 'APPROVED', 'REVIEWED', 'REASON', 'TRIP COMMENTS', 
             'SPECIES COMMENTS','GLOBAL UNIQUE IDENTIFIER']
    return headers


def monthsofyear():
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    return months

def get_list_of_hotspot(hotspot_path):
    hostspot_id_list = []
    hotspot_csv_files = fnmatch.filter(os.listdir(hotspot_path), "*.csv")
    
    for file in hotspot_csv_files:
        filename = file.strip(".csv")
        hostspot_id_list.append(filename)
        
    return hostspot_id_list