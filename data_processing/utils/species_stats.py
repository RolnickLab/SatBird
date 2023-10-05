"""
utility functions to get information about species and their frequencies
"""
import json
import os.path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_species_frequencies(root_dir, output_file="species_frequencies.npy", df_file_name="all_summer_hotspots.csv", num_species=670):
    """
    computes frequency of all species given a csv of hotspots
    """
    df = pd.read_csv(os.path.join(root_dir, df_file_name))

    list_of_frequencies = np.zeros(num_species)

    for i, row in tqdm(df.iterrows()):
        species = json.load(open(os.path.join(root_dir, "corrected_targets", row["hotspot_id"] + '.json')))
        species_probs = np.array(species["probs"])

        species_probs = species_probs * row["num_complete_checklists"]
        list_of_frequencies += species_probs

    print(list_of_frequencies)
    np.save(os.path.join(root_dir, output_file), np.array(list_of_frequencies))


def get_subset_of_species(list_of_frequencies, n=20, output_file="random_subset.npy"):
    """
    get a subset of species given n
    must include the least common species and most common species
    """
    most_common_index = get_most_common_species(list_of_frequencies)
    least_common_index = get_least_common_species(list_of_frequencies)

    list_of_species = np.zeros(n)
    list_of_species[0] = most_common_index
    list_of_species[1] = least_common_index

    list_of_species[2:] = np.random.choice(np.nonzero(list_of_frequencies)[0], n - 2, replace=False)

    np.save(os.path.join(root_dir, output_file), list_of_species)


def get_most_common_species(species_frequencies, output_file=None):
    """
    return index of most common species
    """
    index = np.argmax(species_frequencies)
    if output_file:
        np.save(os.path.join(root_dir, output_file), np.array([index]))
    return index


def get_least_common_species(species_frequencies, output_file=None):
    """
    return index of least_common_species
    """
    index = np.argmin(species_frequencies[np.nonzero(species_frequencies)])
    if output_file:
        np.save(os.path.join(root_dir, output_file), np.array([index]))
    return index


def find_zero_occurance_species(root_dir, summer_species='USA_summer/species_frequencies_updated.npy',
                                winter_species='USA_winter/species_frequencies_updated.npy',
                                outfile='USA_summer/missing_species_updated.npy'):
    summer_freq = np.load(os.path.join(root_dir, summer_species))
    winter_freq = np.load(os.path.join(root_dir, winter_species))

    summer_zero_indices = np.where(summer_freq == 0)[0]
    print("Species with 0 frequency in summer: ", len(summer_zero_indices))
    winter_zero_indices = np.where(winter_freq == 0)[0]
    print("Species with 0 frequency in winter: ", len(winter_zero_indices))
    missing_species = np.intersect1d(winter_zero_indices, summer_zero_indices)
    print("Number of missing species: ", len(missing_species))
    np.save(os.path.join(root_dir, outfile), missing_species)


if __name__ == "__main__":
    root_dir = "/network/projects/ecosystem-embeddings/SatBird_data_v2"
    species_freq_file_name = "species_frequencies_updated.npy"
    # compute_species_frequencies(root_dir, species_freq_file_name)
    find_zero_occurance_species(root_dir)