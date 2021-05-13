#!/usr/bin/env python
# coding: utf-8

# ## **NOTEBOOK: CREATE BAR CHART FOR HOSPOTS ON MONTHLY BASIS**

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ### Read few rows from the CSV file
# CSV File contains ebird information of all localities where number of complete checklist >=50

# - **Note: Right now we only use 500 rows from the hotspot so it will run in seconds but it takes more time when whole file is read (defintely not seconds)**
# - **Question: Can this be made much faster ?**

from pandas.io.common import EmptyDataError
import pandas as pd
import numpy as np
import warnings
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

def main():
    # This file has info of only those hotspots where number of checklist is greater than 50
    csv_file = '/miniscratch/tengmeli/ecosystem-embedding/data/usa_hotspot_data_2.csv'
    file = pd.read_csv(csv_file)

    # Remove CSV file if it already exists
    # Running this cell take a long time so need to work on how to do it. 

    hotspot = 'L197353'
    output_path = str(hotspot) +'_data.csv'
    print("Output file: ", output_path)

    try:
        os.remove(output_path)
    except OSError:
        pass

    # EBIRD DATA FILE LOCATION

    base_folder = '/miniscratch/tengmeli/'
    data = base_folder + 'ecosystem-embedding/data/usa_hotspot_data_2.csv'

    reader = pd.read_csv(data , 
                         chunksize = 1000
                        )
    
    # Read chunks and save to a new csv
    for i,chunk in enumerate(reader):
        if hotspot in chunk['LOCALITY ID'].values:
            print("HF..", end ='')
            state_bird_all = chunk.loc[chunk['LOCALITY ID'] == 'L197353']
            state_bird_all.to_csv(output_path ,mode='a', header=not os.path.exists(output_path))   
            # Progress Bar
            if (i% 1000 == 0):
                print("#", end ='')
    
    print(" Complete reading the file in chunks..Reading hotspot info now......")
    state_bird_all_info = pd.read_csv(output_path, delimiter= ',')

    # ### Read ABA checklist file
    aba_csv = 'ABA_Checklist-8.0.8.csv'
    aba_file = pd.read_csv(aba_csv, delimiter=',', names=["TYPE", "COMMON-NAME", "FRENCH-NAME", "ENGLISH-NAME", "CODE", "CODE-NUMBER"])
    aba_file.fillna('', inplace=True)


    # #### Filter species with Code 1 and 2 in ABA checklist
    # Save the files as `aba_twos_and_ones.csv`
    aba_twos_and_ones = aba_file[((aba_file["CODE-NUMBER"] == 1.0) | (aba_file["CODE-NUMBER"] == 2.0))]
    aba_twos_and_ones.to_csv('aba_twos_and_ones.csv', index = False)


    # #### Get the `COMMON-NAME` as list
    aba_common_name_list = aba_twos_and_ones['COMMON-NAME'].tolist()


    # ### Test for a single hotspot
    # Hotspot Example: L109542. Note: We are only chosing this hotspot information from first 500 line sof the ebird csv file
    hotspot = 'L109542'
    state_bird_all_info = file.loc[file['LOCALITY ID'] == hotspot] #L197353

    # ### Select only the values which are a part of aba_common_name list (i.e. code 1 and code2)
    state_bird_info = state_bird_all_info[state_bird_all_info["COMMON NAME"].isin(aba_common_name_list)]


    # ### Extract months from the date and Time (to be used in bar chart)
    state_bird_info['LAST EDITED DATE'] = pd.to_datetime(state_bird_info['LAST EDITED DATE'])
    state_bird_info['Month'] = state_bird_info['LAST EDITED DATE'].dt.month_name()

    if 'MONTH' not in state_bird_info:
        state_bird_info.insert(state_bird_info.columns.get_loc('LAST EDITED DATE'), 'MONTH', state_bird_info['Month'] )


    # * To verify if `MONTH` column has correct columns use : `state_bird_month[state_bird_month["MONTH"].apply(lambda x:x not in months)]`
    # * To verify if `['OBSERVATION COUNT']` has only numeric values, ue: `state_bird_month.loc[~state_bird_month['OBSERVATION COUNT'].astype(str).str.isdigit()]`
    # * To verfy if any column has NaN values, use something like: `state_bird_month['COMMON NAME'].isnull().values.any()`

    # ### Filter sected column which will ne needed to make bar chart
    # Columns needed are: "LOCALITY ID", "OBSERVATION COUNT", "COMMON NAME", "MONTH", "ALL SPECIES REPORTED"
    state_bird_month = state_bird_info[["LOCALITY ID", "OBSERVATION COUNT", "COMMON NAME", "MONTH", "ALL SPECIES REPORTED"]]
    state_bird_month = state_bird_month[state_bird_month["ALL SPECIES REPORTED"] == 1]


    # ### Sort the filtered data with respect to column
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    state_bird_month['MONTH'] = pd.Categorical(state_bird_month['MONTH'], categories=months, ordered=True)
    state_bird_month.sort_values(by='MONTH',inplace=True) # Sort from Jan - December
    state_bird_month['MONTH'] = state_bird_month['MONTH'].astype(object)
    state_bird_month.head()


    # ### Verify the values are correct after filteration
    old_df = state_bird_info[(state_bird_info["COMMON NAME"] == "Caspian Tern") & 
                    (state_bird_info["MONTH"] == "January")
                            ]
    new_df = state_bird_month[(state_bird_month["COMMON NAME"] == "Caspian Tern") & 
                    (state_bird_month["MONTH"] == "January")
                             ]

    print("Old DF:", old_df[["OBSERVATION COUNT", "ALL SPECIES REPORTED"]])
    print("New DF:", new_df[["OBSERVATION COUNT", "ALL SPECIES REPORTED"]])


    # ### Map the months (string) with numbers for proper sorting 
    months_dict = {"January":1, "February":2, "March":3, "April":4, "May":5, "June":6,
              "July":7, "August":8, "September":9, "October":10, "November":11, "December":12
             }
    state_bird_month_num = state_bird_month.MONTH.map(months_dict)

    if 'Month_Num' not in state_bird_month:
        state_bird_month.insert(0, "Month_Num", state_bird_month_num)
    state_bird_month.head()


    # ### Create bar chart data
    state_bird_month_sum = state_bird_month.groupby(['Month_Num','MONTH','COMMON NAME'])[['ALL SPECIES REPORTED']].agg('sum')
    state_bird_month_sum


    # ### Plot bar Chart
    df_temp = state_bird_month_sum.pivot_table("ALL SPECIES REPORTED", "COMMON NAME", "Month_Num")

    plt.figure(figsize = (25,15))
    plt.title("BAR CHART", size=20)
    # sns.light_palette("seagreen", as_cmap=True)
    ax = sns.heatmap(df_temp, annot = True, cmap = "Greens")
    ax.set_xticklabels(months, rotation='horizontal', fontsize=10)
    plt.savefig("bar_chart_" + hotspot + ".png")


if __name__=="__main__":
    main()


