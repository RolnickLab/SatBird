import pandas as pd
import os

### CREATE CSV WITH HOTSPOT INFORMATION

if __name__ =="__main__":
    root = "/miniscratch/srishtiy/hotspot_csv_data/"
    rows_list = []
    l = [os.path.join(root, i) for i in os.listdir(root)]
    df = pd.DataFrame()
    df["hotspot_id"] = ""
    df["lon"] = ""
    df["lat"] = ""

    for elem in l:
        hs = os.path.basename(elem).strip(".csv")
        print(hs)
        temp = pd.read_csv(elem)#.loc[0]
        temp0 = temp.loc[0]
        lon = temp0["LONGITUDE"]
        lat = temp0["LATITUDE"]
        county = temp0["COUNTY"]
        county_code = temp0["COUNTY CODE"]
        state = temp0["STATE"]
        state_code = temp0["STATE CODE"]
        temp = temp[temp["CATEGORY"]=="species"]
        tmp = temp["SCIENTIFIC NAME"].nunique()
        num_different_species = tmp
        num_checklists = temp["SAMPLING EVENT IDENTIFIER"].nunique()
        num_complete_checklists = temp[temp["ALL SPECIES REPORTED"]==1]["SAMPLING EVENT IDENTIFIER"].nunique()
    
        dict1 = {"hotspot_id": hs,
            "lon":lon,
            "lat":lat,
            "county":county,
            "county_code":county_code,
             "state": state,
             "state_code":state_code,
             
             "num_checklists":num_checklists,
             "num_complete_checklists": num_complete_checklists,
             "num_different_species":num_different_species
             
            }


        rows_list.append(dict1)


    hs_df = pd.DataFrame(rows_list)               


    hs_df.to_csv("hotspots_data.csv")

