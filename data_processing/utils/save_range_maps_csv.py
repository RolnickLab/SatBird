import os
import pickle
import pandas as pd


def save_range_maps_csv(save_path):
    range_path = "/network/projects/ecosystem-embeddings/ebird_dataset/USA_summer/range_maps/"

    files = os.listdir(range_path)

    with open("/network/projects/_groups/ecosystem-embeddings/species_splits/species_list.txt", "r") as f:
        species = [line.strip("\n") for line in f.readlines()]

    range_map_csv = pd.DataFrame(columns = species)

    for file in files:
        data = pickle.load(open(os.path.join(range_path, file), "rb"))
        key= list(data.keys())[-1]
        range_map_csv = range_map_csv.append(data[key], ignore_index=True)
    range_map_csv.to_csv(save_path)

if __name__=="__main__":
    save_range_maps_csv("/network/projects/ecosystem-embeddings/ebird_dataset/USA_summer/range_maps_species.csv")