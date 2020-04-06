""" ##################### Main script #########################

This script has been designed to perform training with several 
variations of the models of the nnUNet framework, including a 
3-fold cross validation of our results. The structure of the 
script is as follows:

    1 - Selection of the training, validation and testing sets.
    This is done for each fold of the cross validation.
    2 - Generation of the .json file, necessary to perform 
    training and validation.
    3 - Preprocessing of the dataset.
    4 - Training routine.
    5 - Inference and evaluation of results.
    6 - At the end of the cross-validation, we compute evalua-
    tion metrics across all folds.

We repeat these steps for each of the modifications that we add
to the nnUNet framewrok. This includes changes in (1) data aug-
mentation, (2) resolution of dataset, and (3) network depth.

"""

# Imports

import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np

import os
import glob
import shutil
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

import nibabel as nib
from nibabel.testing import data_path

import json

from evaluation_metrics import eval_metrics

print(" ")
print("Model optimization for nnunet (Isensee et al. 2019) for segmentation of 3D CTA images")
print("By Pere Canals (2020)")
print(" ")

####################### File management #######################

print("Starting file management setup...")

# Define paths to the database folders (images and labels)

# nnunet_dir = "/Users/pere/opt/anaconda3/envs/nnunet_env/nnUNet/nnunet"
nnunet_dir = "/home/perecanals/nnunet_env/nnUNet/nnunet"

path_images_base = join(nnunet_dir, "nnUNet_base/nnUNet_raw/Task00_grid/database_images")
path_labels_base = join(nnunet_dir, "nnUNet_base/nnUNet_raw/Task00_grid/database_labels")

path_imagesTr = join(nnunet_dir, "nnUNet_base/nnUNet_raw/Task00_grid/imagesTr")
path_labelsTr = join(nnunet_dir, "nnUNet_base/nnUNet_raw/Task00_grid/labelsTr")
path_imagesTs = join(nnunet_dir, "nnUNet_base/nnUNet_raw/Task00_grid/imagesTs")

path_imagesTest = join(nnunet_dir, "inference_test/input")
path_labelsTest = join(nnunet_dir, "inference_test/labels")
path_outputsTest = join(nnunet_dir, "inference_test/outputs")

path_imagesTr_rm = path_imagesTr + "/*"
path_labelsTr_rm = path_labelsTr + "/*"
path_imagesTs_rm = path_imagesTs + "/*"

path_imagesTest_rm = path_imagesTest + "/*.gz"
path_labelsTest_rm = path_labelsTest + "/*"
path_outputsTest_rm = path_outputsTest + "/*"

imagesTr = "./imagesTr/"
labelsTr = "./labelsTr/"
imagesTs = "./imagesTs/"

path_model = join(nnunet_dir, "nnUNet_base/nnUNet_training_output_dir/3d_fullres/Task00_grid/nnUNetTrainer__nnUNetPlans/all")
path_save_model = join(nnunet_dir, "models")

path_inference = path_imagesTest + "/inf"

# Create new directory for the model

model_name = "model_1_default_model"

model_dir = join(path_save_model, model_name)

maybe_mkdir_p(model_dir)

dir_images_base = os.fsencode(path_images_base)
dir_labels_base = os.fsencode(path_labels_base)

# List all available images and labels

list_images_base = []; list_labels_base = []

for file in os.listdir(dir_images_base):
    filename = os.fsdecode(file)
    if filename.endswith(".gz"):
        list_images_base.append(filename)
        continue
    else:
        continue

for file in os.listdir(dir_labels_base):
    filename = os.fsdecode(file)
    if filename.endswith(".gz"):
        list_labels_base.append(filename)
        continue
    else:
        continue

list_images_base.sort()
list_labels_base.sort()

# We attribute some percentages of the database to training, validation and testing

tr_prop   = 0.80
val_prop  = 0.00 # Training set includes validation (80:20)
test_prop = 0.20

samp_tr   = int(np.round(tr_prop   * len(list_images_base)))
samp_val  = int(np.round(val_prop  * len(list_images_base)))
samp_test = int(np.round(test_prop * len(list_images_base)))

while samp_tr + samp_val + samp_test > len(list_images_base):
    samp_test += -1

# We generate an order vector to shuffle the samples before each fold for the cross validation
    
order = np.arange(len(list_images_base))
np.random.shuffle(order)

list_images_base_fold = [list_images_base[i] for i in order]
list_labels_base_fold = [list_labels_base[i] for i in order]

print("check!")
print(" ")
print("Starting 3-fold cross validation")

for i in range(1): # 3-fold cross validation

    print("     Fold", str(i))
    print("         File management...")

    # Remove all files from previous fold

    # files = glob.glob(path_imagesTr_rm)
    # for f in files: os.remove(f)

    # files = glob.glob(path_labelsTr_rm)
    # for f in files: os.remove(f)

    # files = glob.glob(path_imagesTs_rm)
    # for f in files: os.remove(f)

    files = glob.glob(path_imagesTest_rm)
    for f in files: os.remove(f)

    files = glob.glob(path_labelsTest_rm)
    for f in files: os.remove(f)

    files = glob.glob(path_outputsTest_rm)
    for f in files: os.remove(f)

    # We generate an order vector to shuffle the samples before each fold for the cross validation

    list_imagesTr   = list_images_base_fold[0: samp_tr]
    list_labelsTr   = list_labels_base_fold[0: samp_tr]

    list_imagesTs   = list_images_base_fold[samp_tr: samp_tr + samp_val]
    list_labelsTs   = list_labels_base_fold[samp_tr: samp_tr + samp_val]

    list_imagesTest = list_images_base_fold[samp_tr + samp_val: samp_tr + samp_val + samp_test]
    list_labelsTest = list_labels_base_fold[samp_tr + samp_val: samp_tr + samp_val + samp_test]

    # Shift values in order for next fold of cross validation (a shift of samp_test)

    order = np.append(order[samp_test:], order[0:samp_test])

    # Copy all corresponding files of present fold

    # for ii in range(len(list_imagesTr)):
    #     shutil.copyfile(path_images_base + "/" + list_imagesTr[ii],   path_imagesTr   + "/" + list_imagesTr[ii])

    # for ii in range(len(list_labelsTr)):
    #     shutil.copyfile(path_labels_base + "/" + list_labelsTr[ii],   path_labelsTr   + "/" + list_labelsTr[ii])

    # for ii in range(len(list_imagesTs)):
    #     shutil.copyfile(path_images_base + "/" + list_imagesTs[ii],   path_imagesTs   + "/" + list_imagesTs[ii])

    for ii in range(len(list_imagesTest)):
        shutil.copyfile(path_images_base + "/" + list_imagesTest[ii], path_imagesTest + "/" + list_imagesTest[ii])

    for ii in range(len(list_labelsTest)):
        shutil.copyfile(path_labels_base + "/" + list_labelsTest[ii], path_labelsTest + "/" + list_labelsTest[ii])

    # Write the .json file for each fold of the cross validation

    list_imagesTr_json = [None] * len(list_imagesTr)
    list_labelsTr_json = [None] * len(list_labelsTr)
    list_imagesTs_json = [None] * len(list_imagesTs)

    for ii in range(len(list_imagesTr)):
        list_imagesTr_json[ii] = imagesTr + list_imagesTr[ii]
        list_labelsTr_json[ii] = labelsTr + list_labelsTr[ii]

    for ii in range(len(list_imagesTs)):
        list_imagesTs_json[ii] = imagesTs + list_imagesTs[ii]

    dataset = {}

    dataset = {
        "name": "StrokeVessels",
        "description": "Upper Trunk Vessels Segmentation",
        "reference": "Hospital Vall dHebron",
        "licence": "-",
        "release": "1.0 08/01/2020",
        "tensorImageSize": "3D",
        "modality": {
            "0": "CT"
        },
        "labels": {
            "0": "background",
            "1": "vessel"
        },
        "numTraining": samp_tr,
        "numTest": samp_val,
        "training": [],
        "test": []
    }

    # We prepare the training and "testing" samples for the json file

    aux = []
    for ii in range(len(list_imagesTr_json)):
        aux = np.append(aux, {
                        "image": list_imagesTr_json[ii],
                        "label": list_labelsTr_json[ii]
                    })

    aux = aux.tolist()

    aux2 = []
    for ii in range(len(list_imagesTs_json)):
        aux2 = np.append(aux2, list_imagesTs_json[ii])

    if len(aux2) > 0:
        aux2 = aux2.tolist()

    dataset["training"] = aux
    dataset["test"] = aux2

    with open('dataset.json', 'w') as outfile:
        json.dump(dataset, outfile, indent=4)

    # Move json file to nnUNet_raw dir

    os.rename(nnunet_dir + "/dataset.json", nnunet_dir + "/nnUNet_base/nnUNet_raw/Task00_grid/dataset.json")
    shutil.copyfile(nnunet_dir + "/nnUNet_base/nnUNet_raw/Task00_grid/dataset.json", nnunet_dir + "/nnUNet_base/nnUNet_preprocessed/Task00_grid/dataset.json")
    shutil.copyfile(nnunet_dir + "/nnUNet_base/nnUNet_raw/Task00_grid/dataset.json", nnunet_dir + "/nnUNet_base/nnUNet_raw_cropped/Task00_grid/dataset.json")
    shutil.copyfile(nnunet_dir + "/nnUNet_base/nnUNet_raw/Task00_grid/dataset.json", nnunet_dir + "/nnUNet_base/nnUNet_raw_splitted/Task00_grid/dataset.json")

    print("         check!")
    print(" ")

    # With the json file ready, we shall begin preprocessing first:

    ############### Plan and preprocessing ############      Preprocessing done in a separate server with very powerful CPU

    # print("         Preprocessing...")

    # os.system("python3 experiment_planning/plan_and_preprocess_task.py -t Task00_grid -pl 4 -pf 4")

    # print("         check!")

    ##################### Training ####################

    print("         Training...")

    os.system("OMP_NUM_THREADS=1 python3 run/run_training.py 3d_fullres nnUNetTrainer Task00_grid all --ndet")

    print("         check!")

    ##################### Inference ################### Perform inference one image at the time? Memory problems

    print("         Inference...")

    #os.remove(path_model + "/model_best.model")
    #shutil.move(nnunet_dir + "/models/model_1_default_model/model_best.model", path_model + "/") # remove this

    maybe_mkdir_p(path_inference)

    for image in list_imagesTest:
        shutil.move(path_imagesTest + "/" + image, path_inference + "/" + image)

        os.system("OMP_NUM_THREADS=1 python3 inference/predict_simple.py -i " + path_inference + " -o " + path_outputsTest + " -t Task00_grid -tr nnUNetTrainer -m 3d_fullres -f all")

        shutil.move(path_inference + "/" + image, path_imagesTest + "/" + image)
    
    print("         check!")
    print(" ")

    # Perform testing over inferred samples

    print("         Evaluation metrics...")

    eval_met = eval_metrics()
    acc_mean, acc_std, sen_mean, sen_std, spe_mean, spe_std, dice_mean, dice_std = eval_met

    # Create new file for the fold

    fold_dir = join(model_dir, "fold" + str(i))
    maybe_mkdir_p(fold_dir)

    # Move outputs to fold

    shutil.move(path_outputsTest, fold_dir)

    # Move dataset.json to fold

    shutil.move(nnunet_dir + "/nnUNet_base/nnUNet_raw/Task00_grid/dataset.json", fold_dir)

    # Save model file and files for each fold and model

    for file in os.listdir(path_model):
        filename = os.fsdecode(file)
        os.rename(path_model + "/" + filename, fold_dir + "/" + filename)

    eval_file = join(fold_dir, "eval_metrics.csv")
    np.savetxt(eval_file, eval_met)

    print("Fold", str(i), "for model", model_name, "complete. Evaluation metrics:")
    print("accuracy =", acc_mean, acc_std)
    print("sensitivity =", sen_mean, sen_std)
    print("specificity =", spe_mean, spe_std)
    print("dice score =", dice_mean, dice_std)

    print("         check!")
    print(" ")

    print("     End of fold", str(i))

    print("###################################")