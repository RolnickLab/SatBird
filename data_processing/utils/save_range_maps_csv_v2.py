import os
import pickle
import pandas as pd
from tqdm import tqdm
import json
import argparse

def save_range_maps_csv(save_path, range_path= "/network/projects/ecosystem-embeddings/ebird_dataset/USA_summer/range_maps/"):
    """
    * save range maps in csv format with all species for which we have range maps there.
      
    """

    files = os.listdir(range_path)

    # with open("/network/projects/_groups/ecosystem-embeddings/species_splits/species_list.txt", "r") as f:
    #     species = [line.strip("\n") for line in f.readlines()]
    
    data_list = []
    for file in tqdm(files):
        data = pickle.load(open(os.path.join(range_path, file), "rb"))
        if "loc" in list(data.keys()):
            key= list(data.keys())[-1]
            data = data[key]
        data["hotspot_id"] = file.strip(".pkl")
        data_list.append(data)

    range_map_csv = pd.DataFrame(data_list)
    
    
    with open("/network/projects/_groups/ecosystem-embeddings/species_splits/species_list.txt", "r") as f:
         species = [line.strip("\n") for line in f.readlines()]
    
    #add columns of species for which we don't have range maps 
    s = [spe for spe in species if spe not in range_map_csv.columns]
    range_map_csv[s]=True
    range_map_csv = range_map_csv.fillna(True)
    range_map_csv = range_map_csv[species + ["hotspot_id"]]
    
    range_map_csv.to_csv(save_path)
    
def correct_targets(rm_csv, 
                    outpath = "/network/projects/ecosystem-embeddings/ebird_dataset/USA_summer/corrected_targets/",
                    targets_path = "/network/projects/ecosystem-embeddings/ebird_dataset/USA_summer/targets/"):
    """
    *csv with range maps masks for species for each location obtained with save_range_maps_csv
    * do not save correct targets of hotspots that are already in the output path
    """
    csv = pd.read_csv(rm_csv, index_col="hotspot_id")
    csv = csv.loc[:,~csv.columns.str.startswith('Unnamed')]
 
    for i,row in csv.iterrows():
        if os.path.exists(os.path.join(outpath, f"{i}.json")):
            continue
        with open(os.path.join(targets_path, f"{i}.json"), "rb") as f:
            data= json.load(f)
        data["probs"] = list(data["probs"] * row.values)
        with open(os.path.join(outpath, f"{i}.json"), "w") as fp:
            json.dump(data, fp)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path_csv",
        type=str,
        help="path where to save the csv of range maps.",
    )
    parser.add_argument(
        "--rm_path",
        type=str,
        default="/network/projects/ecosystem-embeddings/ebird_dataset/USA_summer/range_maps/",
        help="path to original range maps folder.",
    )
    parser.add_argument(
        "--outpath_corrected_targets",
        type=str,
        default="/network/projects/ecosystem-embeddings/ebird_dataset/USA_summer/corrected_targets/",
        help="path where to save corrected targets.",
    )
    parser.add_argument(
        "--targets_path",
        type=str,
        default="/network/projects/ecosystem-embeddings/ebird_dataset/USA_summer/targets/",
        help="path to original targets folder.",
    )
    
    
    args = parser.parse_args()
    save_range_maps_csv(args.save_path_csv, args.rm_path)
    
    correct_targets(args.save_path_csv, 
                    outpath = args.outpath_corrected_targets,
                    targets_path = args.targets_path)
    print("Done")
    
if __name__=="__main__":

    main()