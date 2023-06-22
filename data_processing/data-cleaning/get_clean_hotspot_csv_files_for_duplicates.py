import pandas as pd
import os
from os import listdir
from os.path import isfile, join


def remove_row_duplicates(mypath, output_csv_folder, onlyfiles):
    """Removes duplicate rows from the csv files in source destination and saves them to output destination
    """
    for idx, file in enumerate(onlyfiles):
        # Absolute path of pre-processed csv file
        csv_path = mypath + file

        # Absolute path of post-processed csv file
        output_path = output_csv_folder + file

        # Convert csv to dataframe
        single_csv_df = pd.read_csv(csv_path, delimiter=",")

        # Drop all duplicate rows
        no_duplicates = single_csv_df.drop_duplicates()

        # One single row is still same as header (because its unique row). Remove it
        single_csv_cleaned_df = no_duplicates.loc[no_duplicates['GLOBAL UNIQUE IDENTIFIER']!='GLOBAL UNIQUE IDENTIFIER']

        single_csv_cleaned_df.to_csv(output_path, index = False)

        if (idx % 100 == 0):
            print("#", end ='')
            
    
def main():
    # Path of all individual CSV files for each hotspot
    src_csv_folder = ''
    dest_csv_folder = ''

    # List of all such csv files
    onlyfiles = [f for f in listdir(src_csv_folder) if isfile(join(src_csv_folder, f))]
    
    remove_row_duplicates(src_csv_folder, dest_csv_folder, onlyfiles)

if __name__ == "__main__":
    main()