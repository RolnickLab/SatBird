import pandas as pd
import os
import json
import tqdm
import numpy as np
from pathlib import Path

def merge_target(outpath, json_dict="/network/projects/ecosystem-embeddings/ebird_new/to_merge_dict.json", root = "/network/projects/ecosystem-embeddings/ebird_new/summer_targets/"):
    """
    json_dict = dictionary to be used to  merge targets from hotspots that have duplicate lat lons: {hotspot_in_the_hotspot_csv: [duplicate_lat_lon_hotspots]}
    root = path to root of targets
    """
    root = Path(root)
    with open(json_dict, "r") as f:
        data = json.load(f)
        
    for key in data:
        #if not os.path.exists(root /f"{key}.json"):
        paths = [root /f"{key}.json"]
        paths += [root / f"{val}.json" for val in data[key]]
        num_checklists = 0
        probs = np.zeros((684))
        for elem in paths:
            if not os.path.exists(elem):
                continue
            with elem.open(mode="r", encoding="utf-8") as u:
                info = json.load(u)
            num_checklists += info['num_complete_checklists']
            probs += np.array(info["probs"])*info['num_complete_checklists']
        if num_checklists==0:
            continue
        info["hotspot_id"]=key
        info["probs"] = list(probs/num_checklists)
        info["num_complete_checklists"] = num_checklists
        
        with open(os.path.join(outpath, f"{key}.json"), 'w') as fp:
            json.dump(info, fp)
if __name__ == "__main__":
     #merge_target(outpath = "/network/projects/ecosystem-embeddings/ebird_new/summer_targets_merged/", 
     #             json_dict="/network/projects/ecosystem-embeddings/ebird_new/to_merge_dict.json", 
     #             root = "/network/projects/ecosystem-embeddings/ebird_new/summer_targets/")
    merge_target(outpath = "/network/projects/ecosystem-embeddings/ebird_new_targets/winter_targets_merged/", 
                  json_dict="/network/projects/ecosystem-embeddings/ebird_new/to_merge_dict.json", 
                  root = "/network/projects/ecosystem-embeddings/ebird_new_targets/winter_targets/")