"""
Save ebird data complete checklists information as a pickle
"""

import pandas as pd

base_folder = '/miniscratch/srishtiy/'

# complete checklists file for full ebird data 
data_checklists = base_folder + 'complete-checklists_Jan2021.txt'
df = pd.read_csv(data_checklists , delimiter = "\t", keep_default_na=False)

# Save complete checklists for whole ebird data In pickle
complete_checklist_path_pkl = base_folder + 'complete_checklist_Jan2021_full_ebird.pkl'
df.to_pickle(complete_checklist_path_pkl)