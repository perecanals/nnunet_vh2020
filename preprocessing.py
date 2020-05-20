import numpy as np

import os
from glob import glob
import shutil
import json

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

def preprocessing(nnunet_dir):
    ''' ############################## Preprocessing #########################################

    Preprocessing of the whole dataset as described by Isensee et al. (2019).

    Images (labels) should be in the database_images/ (database_labels/) 
    dir inside: ./nnunet/nnUNet_base/nnUNet_raw/Task00_grid/ 

    Git repository: https://github.com/perecanals/nnunet_vh2020.git
    Original nnunet (Isensee et al. 2020[1]): https://github.com/MIC-DKFZ/nnUNet.git

    [1] Fabian Isensee, Paul F. Jäger, Simon A. A. Kohl, Jens Petersen, Klaus H. Maier-Hein "Automated Design of Deep Learning 
    Methods for Biomedical Image Segmentation" arXiv preprint arXiv:1904.08128 (2020).

     '''
    ################################## Preprocessing #########################################

    # Paths
    path_images_base = os.path.join(nnunet_dir, 'database_vh/database_images')
    path_labels_base = os.path.join(nnunet_dir, 'database_vh/database_labels')

    path_imagesTr = os.path.join(nnunet_dir, 'nnUNet_base/nnUNet_raw_data/Task100_grid/imagesTr')
    path_labelsTr = os.path.join(nnunet_dir, 'nnUNet_base/nnUNet_raw_data/Task100_grid/labelsTr')
    path_imagesTs = os.path.join(nnunet_dir, 'nnUNet_base/nnUNet_raw_data/Task100_grid/imagesTs')

    imagesTr = './imagesTr/'
    labelsTr = './labelsTr/'

    # List all available images and labels
    dir_images_base = os.fsencode(path_images_base)
    dir_labels_base = os.fsencode(path_labels_base)
    list_images_base = []; list_labels_base = []
    for file in os.listdir(dir_images_base):
        filename = os.fsdecode(file)
        if filename.endswith('.gz'):
            list_images_base.append(filename)
            continue
        else:
            continue
    for file in os.listdir(dir_labels_base):
        filename = os.fsdecode(file)
        if filename.endswith('.gz'):
            list_labels_base.append(filename)
            continue
        else:
            continue

    print('Total number of images available in database:', len(list_images_base))
    print('                                                                    ')

    # Remove all preexisting nifti files
    print('Removing possibly preexisting nifti files...')

    for files in glob(os.path.join(path_imagesTr, '*.gz')):
        os.remove(files)
    for files in glob(os.path.join(path_labelsTr, '*.gz')):
        os.remove(files)
    for files in glob(os.path.join(path_imagesTs, '*.gz')):
        os.remove(files)

    print('done')
    print('    ')

    # Copy files to corresponding directories
    print('Copying new files...')

    for image in list_images_base:
        shutil.copyfile(os.path.join(path_images_base, image),   os.path.join(path_imagesTr, image[:8] + '_0000.nii.gz'))
    for label in list_labels_base:
        shutil.copyfile(os.path.join(path_labels_base, label),   os.path.join(path_labelsTr, label))

    print('done')
    print('    ')

    # Write the .json file 
    print('Writing .json file for preprocessing...')

    list_imagesTr_json = [None] * len(list_images_base)
    list_labelsTr_json = [None] * len(list_labels_base)
    list_imagesTs_json = [None]
    for idx, _ in enumerate(list_images_base):
        list_imagesTr_json[idx] = imagesTr + list_images_base[idx]
        list_labelsTr_json[idx] = labelsTr + list_labels_base[idx]

    dataset = {}
    dataset = {
        "name": "Data preprocessing",
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
        "numTraining": len(list_labels_base),
        "numTest": 0,
        "training": [],
        "test": []
    }

    # We prepare the preprocessing samples for the json file
    aux = []
    for idx, _ in enumerate(list_imagesTr_json):
        aux = np.append(aux, {
                        "image": list_imagesTr_json[idx],
                        "label": list_labelsTr_json[idx]
                    })
    aux = aux.tolist()

    aux2 = []
    for idx, _ in enumerate(list_imagesTs_json):
        aux2 = np.append(aux2, list_imagesTs_json[idx])
    if len(aux2) > 0:
        aux2 = aux2.tolist()

    dataset['training'] = aux
    dataset['test'] = aux2

    with open('dataset.json', 'w') as outfile:
        json.dump(dataset, outfile, indent=4)

    print('done')
    print('    ')

    # Move json file to Task00_grid
    print('Moving .json to Task100_grid directory...')

    os.rename(nnunet_dir + "/dataset.json", nnunet_dir + "/nnUNet_base/nnUNet_raw_data/Task100_grid/dataset.json")

    print('done')
    print('    ')

    ############################# Plan and preprocessing #####################################

    print('Starting preprocessing...')

    os.system('nnUNet_plan_and_preprocess -t 100 -tl 16 -tf 8')

    print('Preprocessing finished')



##############################################################################################
#--------------------------------------------------------------------------------------------#    
##############################################################################################