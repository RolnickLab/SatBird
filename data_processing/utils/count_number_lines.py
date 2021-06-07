import os
import pandas as pd

def main():
    #example if you just want to count the lines in one csv
    line = 0
    chksz = 1000
    for chunk in pd.read_csv("/miniscratch/tengmeli/ecosystem-embedding/data/usa_hotspot_data_final.csv",chunksize = chksz,index_col=0, usecols=["CATEGORY"]):
        line += chunk.shape[0]
        print(line)
    return(line)

if __name__=="__main__": 
    #count lines in all csv in a folder
    line = 0
    chksz = 1000
    csv_list = os.listdir("/home/mila/t/tengmeli/ecosystem-embedding/hotspot_all_csv")
    for csv in csv_list:
        tmp_csv = os.path.join("/home/mila/t/tengmeli/ecosystem-embedding/hotspot_all_csv", csv)
        for chunk in pd.read_csv(tmp_csv,chunksize = chksz,index_col=0, usecols=["CATEGORY"]): #typically only use one column to make reading/counting lines faster
        
            line += chunk.shape[0]
            print(line) 
    print(line)
