"""
CREATES BAR CHART FOR SPECIES REPORTED PER MONTH FOR ALL SPECIES IN ABA LIST CDE 1 AND 2
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_processing.utils.helper_data import monthsofyear

# set hotspot id
hotspot_id = 'L452657'

complete_dataframe_sorted_pkl_path  = "/miniscratch/srishtiy/species_object/" + hotspot_id + "_all_aba_hotspot_data.pkl"
df = pd.read_pickle(complete_dataframe_sorted_pkl_path)

# Fill NA with 0 where reported species were not seen (e.g. NaN)
df_month_sum = df.groupby(['Month_Num','MONTH','COMMON NAME'])[['ALL SPECIES REPORTED']].agg('sum').reset_index()
df_month_sum['ALL SPECIES REPORTED'] = df_month_sum['ALL SPECIES REPORTED'].fillna(0)

# Plot and save bar Chart
months =  monthsofyear()
df_temp = df_month_sum.pivot_table("ALL SPECIES REPORTED", "COMMON NAME", "Month_Num")

plt.figure(figsize = (25,40))
plt.title("BAR CHART", size=20)
# sns.light_palette("seagreen", as_cmap=True)
ax = sns.heatmap(df_temp, annot = True, cmap = "Greens")
ax.set_xticklabels(months, rotation='horizontal', fontsize=10)

bar_chart_name = 'images/' + hotspot_id + '_barchart.png'
plt.savefig(bar_chart_name)
