"""
The code create a csv file which has all the hotspots of United States whose 'LOCALITY ID' 
is _________________________ (ask meli)
"""

import os
import pandas as pd
import time

def main():
    base_folder = '/miniscratch/srishtiy/'
    data = base_folder + 'ebd_relJan-2021.txt'

    reader = pd.read_csv(data , 
                         delimiter = "\t",
                         chunksize = 1000
                        )
    
    # TODO: Ask Melisande to update the csv file if needed
    output_path = 'usa_hotspot_data_2.csv'
    print("Output file: ", output_path)

    try:
        os.remove(output_path)
    except OSError:
        pass
    list_loc = []
    
    # TO DO: What is hotspot_ids.txt again? Ask Meli.
    with open("hotspot_ids.txt", 'r') as f:
        list_loc = [current_place.rstrip() for current_place in f.readlines()]
        
    # Read chunks and save to a new csv
    for i,chunk in enumerate(reader):
            timee = time.time()
            usa_chunk = chunk.loc[chunk['COUNTRY'] == 'United States']
            usa_chunk_locality = usa_chunk.loc[usa_chunk['LOCALITY ID'].isin(list_loc)]
            usa_chunk_locality.to_csv(output_path ,mode='a', header=not os.path.exists(output_path))   
            print(i)
            print(time.time() - timee)
                
            
if __name__=="__main__":
    main()
