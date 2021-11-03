# %%
import pydicom
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from ipywidgets import widgets

INSTANCE_NUMBER = 0x00200013

# %%
path = 'P:/biomech-stud/Wilhelm/Phoenix/104540_CD1_3/DICOM/0000A662/AA234E46/AAB8B401/0000F28C'
filenames = os.listdir(path)

slices = []
for loc_filename in tqdm(filenames):
    slices.append(pydicom.read_file(os.path.join(path, loc_filename)))

# %%
slices.sort(key=lambda x: int(x[INSTANCE_NUMBER].value))

# %%


def update(idx=0):
    plt.clf()
    plt.figure(figsize=(10, 10))
    plt.imshow(slices[idx].pixel_array, cmap=plt.cm.bone)
    plt.axis('off')
    plt.show()
    print(int(slices[idx][INSTANCE_NUMBER].value))


widgets.interact(update, idx=(0, len(slices)-1))
# %%