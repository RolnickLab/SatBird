import os
import glob
import json
import pickle
import numpy as np

if __name__ == "__main__":
    paths = glob.glob("/miniscratch/tengmeli/ecosystem-embedding/ebird_data/*complete_checklists.pkl")
    print(len(paths))
    for path in paths:
        print(path)
        id_ = os.path.basename(path).strip("_complete_checklists.pkl")
        file = open(path, "rb")
        pkl = pickle.load(file)
        
        june = pkl[pkl["MONTH"]=="June"]
        complete_checklists = int(june["MONTHWISE_COMPLETE_CHECKLISTS"].values[0])
        if complete_checklists > 0:
            probas = list(june["SPECIES_COMPLETE_CHECKLISTS"]/complete_checklists)
        
            d = {"hotspot_id": id_,
                "probs": probas,
                "num_complete_checklists": complete_checklists}
            outfile= os.path.join("/miniscratch/tengmeli/ecosystem-embedding/ebird_data_june/", id_ +".json")
            with open(outfile, "w")  as json_file:
                json.dump(d, json_file, allow_nan=False)
            json_file.close()
    