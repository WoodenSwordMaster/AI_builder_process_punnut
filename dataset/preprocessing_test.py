import os
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting

data_path = r"E:\test"
out_path = r"C:\Users\punnut\Downloads\test\skull_stripping_test"
for root, dir,files in os.walk(data_path):
    for file in files:
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            nii_path = root + "/" + file
            nii_img = nib.load(nii_path)
            print(file, "==", nii_img.shape)
            fig, ax = plt.subplots(figsize=[10, 5])
            plotting.plot_img(nii_img, cmap='gray', axes=ax, display_mode='mosaic')#, cut_coords=(0, 0, 0))
            plt.show()
