import rasterio as rio
import numpy as np
import os
import matplotlib.pyplot as plt
import tifffile as tiff
from torchvision.transforms import ToTensor

hotspot_id = "L4325647" # self.df.iloc[index]['hotspot_id']
type = 'refl'
root_dir = '/home/hagerradi/projects/Ecosystem_embeddings/ecosystem-embedding/samples'
if type == 'img':
    sample_path = os.path.join(root_dir, hotspot_id + '_visual.tif')
else:
    sample_path = os.path.join(root_dir, hotspot_id + '.tif')

if type == "refl":
    with rio.open(sample_path) as f:
        nir = f.read(4)     #B8
        r = f.read(3)       # B4
        g = f.read(2)       # B3
        b = f.read(1)       # B2
    composite = np.stack((r, g, b, nir), axis=-1)
    print(composite.shape)
    normalized_composite = np.clip((composite / 10000), 0, 1)

    img = tiff.imread(sample_path)
    new_band_order = [2, 1, 0, 3]
    rearranged_data = img[:, :, new_band_order].astype(np.float)
    rearranged_data[:, :, -1] = (rearranged_data[:, :, -1] / img[:, :, -1].max()) * 255
    rearranged_data = rearranged_data / 255

    print("re: ", rearranged_data.shape)
    image = ToTensor()(rearranged_data)
    print(image.size())


if type == "img":
    img = tiff.imread(sample_path)
    image = ToTensor()(img)

    plt.imshow(img)