import os
import nibabel as nib
from scipy import ndimage
import numpy as np
from numpy import random
import torch
import matplotlib.pyplot as plt
import torchio as tio
#from nilearn import image
import nilearn
from nilearn import maskers

def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

# def normalize(volume):
#     """Normalize the volume"""
#     min = -1000
#     max = 400
#     volume[volume < min] = min
#     volume[volume > max] = max
#     volume = (volume - min) / (max - min)
#     volume = volume.astype("float32")
#     return volume

def normalize2(volume):
    # Normalize the volume to zero mean and unit variance
    volume = (volume - np.mean(volume)) / np.std(volume)  # <-- Added normalization
    volume = volume.astype("float32")
    return volume

def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 64
    desired_height = 64
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

# def random_crop(data, save_path, save_path2):
#     from nilearn.masking import apply_mask
#     number = random.randint(10 + 1)
#     all_depth = data.shape[-1]
#     all_width = data.shape[0]
#     all_height = data.shape[1]
#     size = 8
#     img = np.zeros((all_height,all_width,all_depth,1), np.uint8)
#     img.fill(1)
#     fill_image = np.zeros((size,size,size,1), np.uint8)
#     #fill_image.fill(1)
#     # print("data",data.shape)
#     # print("fill_image",fill_image.shape)
#     data = np.expand_dims(data, axis=-1)
#     for i in range(1, number + 1):
#         # print("depth1", all_depth, all_depth - size + 1)
#         # print('width1', all_width, all_width - size + 1)
#         # print('height1', all_height, all_height - size + 1)
#         depth_point = random.randint(all_depth - size + 1)
#         width_point = random.randint(all_width - size + 1)
#         height_point = random.randint(all_height - size + 1)
#         img[height_point:height_point+size, width_point:width_point+size, depth_point:depth_point+size] = fill_image
#
#     data2 = nib.Nifti1Image(data, affine=np.eye(4))
#     nib.save(data2, save_path)
#     data2 = nilearn.image.load_img(save_path)
#
#     img2 = nib.Nifti1Image(img, affine=np.eye(4))
#     nib.save(img2, save_path2)
#     img2 = nilearn.image.load_img(save_path2)
#
#     # print(data2.shape)
#     # print(img2.shape)
#     masker = maskers.NiftiMasker(mask_img=img2)
#     data3 = masker.fit_transform(data2)
#     data3 = masker.inverse_transform(data3)
#     return data3

def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize2(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume


data_path = r"C:\Users\punnut\Downloads\final_dataset1"
final_path = r"C:\Users\punnut\Downloads\final_dataset1_preprocessed2_traintest_64"
AD_class = []
CN_class = []
MCI_class = []
MCI_AD_class = []
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".nii"):
            pathList = root.split(os.sep)
            for i in pathList:
                if i == "AD":
                    AD_class.append(os.path.join(root,file))
                    #print(os.path.join(root,file))
                elif i == "CN":
                    CN_class.append(os.path.join(root, file))
                    #print(os.path.join(root, file))
                elif i == "MCI":
                    MCI_class.append(os.path.join(root, file))
                    #print(os.path.join(root, file))
                elif i == "MCI_AD":
                    MCI_AD_class.append(os.path.join(root, file))
                    #print(os.path.join(root, file))

from sklearn.model_selection import train_test_split
all_list = []
if len(AD_class) != 0:
    all_list.append(AD_class)
if len(MCI_class) != 0:
    all_list.append(MCI_class)
if len(MCI_AD_class) != 0:
    all_list.append(MCI_AD_class)
if len(CN_class) != 0:
    all_list.append(CN_class)

for list in all_list:
    split_list = []
    for path in list:
        file = process_scan(path)
        print(path,"processed")
        array = [file,path]
        split_list.append(array)
    train, val = train_test_split(split_list, test_size=0.2, random_state=42)
    print("finished split train val")

    for array in train:
        new_path = array[1].replace(data_path,final_path + "/train")
        if new_path.find("CN_I") != -1:
            string_list = new_path.partition("CN_I")
            new_path = string_list[0]
            file = string_list[1] + string_list[2]
        elif new_path.find("MCI_I") != -1:
            string_list = new_path.partition("MCI_I")
            new_path = string_list[0]
            file = string_list[1] + string_list[2]
        elif new_path.find("MCI_AD_I") != -1:
            string_list = new_path.partition("MCI_AD_I")
            new_path = string_list[0]
            file = string_list[1] + string_list[2]
        elif new_path.find("AD_I") != -1:
            string_list = new_path.partition("AD_I")
            new_path = string_list[0]
            file = string_list[1] + string_list[2]

        data = array[0]
        data = np.expand_dims(data, axis=0)
        preprocess_list = [tio.RandomFlip(axes=('L', 'R', 'P', 'A', 'I', 'S')),
                        tio.RandomAffine(degrees=(90,180,270)),
                        tio.RandomBiasField(coefficients=(0.1,0.16),order=1),
                        tio.RandomElasticDeformation(num_control_points=7,max_displacement=7,locked_borders=1)]
        preprocess_list2 = [tio.RandomGamma(log_gamma=(-0.5, 0.5)),
                        tio.RandomSwap(patch_size=8,num_iterations=15)]
        transform = tio.Compose(preprocess_list2)

        if not os.path.exists(new_path):
            os.makedirs(new_path)

        print("train transforming", os.path.join(new_path, file))
        n = 6
        for i in range(1, n + 1):
            data_n = transform(data)
            data_n = normalize2(data_n)
            data_n = np.reshape(data_n,(64,64,64))
            data_n = np.expand_dims(data_n, axis=-1)
            image_n = nib.Nifti2Image(data_n, affine=np.eye(4))
            nib.save(image_n, os.path.join(new_path, file.replace(".nii", "_" + str(i) +".nii")))
            print("transformed", i)

    for array in val:
        new_path = array[1].replace(data_path, final_path + "/test")
        if new_path.find("CN_I") != -1:
            string_list = new_path.partition("CN_I")
            new_path = string_list[0]
            file = string_list[1] + string_list[2]
        elif new_path.find("MCI_I") != -1:
            string_list = new_path.partition("MCI_I")
            new_path = string_list[0]
            file = string_list[1] + string_list[2]
        elif new_path.find("MCI_AD_I") != -1:
            string_list = new_path.partition("MCI_AD_I")
            new_path = string_list[0]
            file = string_list[1] + string_list[2]
        elif new_path.find("AD_I") != -1:
            string_list = new_path.partition("AD_I")
            new_path = string_list[0]
            file = string_list[1] + string_list[2]

        if not os.path.exists(new_path):
            os.makedirs(new_path)

        data = array[0]
        data = np.expand_dims(data, axis=-1)
        image_n = nib.Nifti2Image(data, affine=np.eye(4))
        nib.save(image_n, os.path.join(new_path, file))
        print("vaild transformed", os.path.join(new_path, file))



# for root, dirs, files in os.walk(data_path):
#     for file in files:
#         if file.endswith(".nii"):
#             file_path = os.path.join(root, file)
#             print(file_path)
#             new_path = root.replace(data_path,final_path)
#
#             data = read_nifti_file(file_path)
#             data = normalize2(data)
#             data = resize_volume(data)
#             #print(data.shape)
#
#             # data = random_crop(data,os.path.join(new_path,file),os.path.join(new_path,file.replace(".nii","_2.nii")))
#             # os.remove(os.path.join(new_path, file.replace(".nii", "_2.nii")))
#             # print(data.shape)
#
#             data = np.expand_dims(data, axis=0)
#             preprocess_list = [tio.RandomFlip(axes=('L', 'R', 'P', 'A', 'I', 'S')),
#                                tio.RandomAffine(degrees=(90,180,270)),
#                                tio.RandomBiasField(coefficients=(0.1,0.16),order=1),
#                                tio.RandomElasticDeformation(num_control_points=7,max_displacement=7,locked_borders=1)]
#             preprocess_list2 = [tio.RandomGamma(log_gamma=(-0.5, 0.5)),
#                                 tio.RandomSwap(patch_size=8,num_iterations=12)]
#             transform = tio.Compose(preprocess_list2)
#             if not os.path.exists(new_path):
#                 os.makedirs(new_path)
#             n = 1
#             for i in range(1, n + 1):
#                 data_n = transform(data)
#                 data_n = normalize2(data_n)
#                 data_n = np.reshape(data_n,(64,64,64))
#                 data_n = np.expand_dims(data_n, axis=-1)
#                 print("transformed",i)
#                 image_n = nib.Nifti2Image(data_n, affine=np.eye(4))
#                 nib.save(image_n, os.path.join(new_path, file.replace(".nii", "_" + str(i) +".nii")))




