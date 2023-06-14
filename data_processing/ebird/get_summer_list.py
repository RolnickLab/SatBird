import os
import numpy as np
from tqdm import tqdm
import pandas as pd
if __name__=="__main__":
    paths = os.listdir("/network/projects/_groups/ecosystem-embeddings/ebird_new/checklists_USA/summer/")
    with open('/network/projects/_groups/ecosystem-embeddings/ebird_new/checklists_USA/summer_list.txt', 'w') as f:
        for file in tqdm(paths):
            i = pd.read_csv(os.path.join("/network/projects/_groups/ecosystem-embeddings/ebird_new/checklists_USA/summer/", file))
            if len(i) > 5:
                f.write( i["LOCALITY ID"][0] + "," + str(i["LATITUDE"][0])+ "," + str(i["LONGITUDE"][0]))
                f.write('\n')
    print("Done")
