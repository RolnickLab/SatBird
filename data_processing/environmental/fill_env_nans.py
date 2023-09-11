import os
import csv
import numpy as np
from scipy.interpolate import interp2d
import glob

rasters = glob.glob("/network/projects/ecosystem-embeddings/SatBird_data_v2/USA_summer/environmental/*")


H, W = 50,50
# Example data with NaN values
x = np.linspace(0, H-1, H)
y = np.linspace(0, W-1, W)
xi, yi = np.meshgrid(x, y)

def bicubic_interpolation(elem, nans):
    # Perform bicubic interpolation
    f = interp2d(xi[~nans], yi[~nans], elem[~nans], kind='cubic', bounds_error=False)
    return f(x, y)

def main():
    for file in rasters: 
        r = np.load(file)
        u = r.copy()
        for i,elem in enumerate(r): 
            if i%1000==0:
                print(i)
            nans = np.isnan(elem)
            nonnan = ~np.isnan(elem)
            if nans.all():
                #all values are nan, need to fill with something
                #TODO
                continue #for now
            elif nans.any() and nonnan.any():
                # Define the coordinates of NaN values
                # Create a function for bicubic interpolation with extrapolation
                arr = bicubic_interpolation(elem, nans)
                arr[~nans] = elem[~nans]
                u[i] = arr

          # save outside the for loop
        np.save(os.path.join("/network/projects/_groups/ecosystem-embeddings/SatBird_data_v2/env_filled/", os.path.basename(file)), u)
        
if __name__=="__main__":
    main()
