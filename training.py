import numpy as np

import os
import glob
import shutil
import json

from time import time

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

def training(nnunet_dir, FOLDS=1, SKIP_FOLD=0):
    ''' ############################### Training #############################################

    Training of the nnunet (3-fold cross validation).

    We perform training for each of the modifications that we add
    to the nnUNet framewrok. This includes (0) default out-of-the-box 
    nnunet, (1) changes in data augmentation, (2) lower resolution 
    of the dataset.

    Preprocessing of the whole dataset and file management should be 
    perform prior to this script.

    Images (labels) should be in the database_images/ (database_labels/) 
    dir inside: ./nnunet/nnUNet_base/nnUNet_raw/Task00_grid/ 

    Git repository: https://github.com/perecanals/nnunet_vh2020.git
    Original nnunet (Isensee et al. 2019): https://github.com/MIC-DKFZ/nnUNet.git

    '''

    # Paths
    path_models = join(nnunet_dir, 'nnUNet_base/nnUNet_training_output_dir/3d_fullres/Task100_grid/nnUNetTrainer__nnUNetPlans')
    path_save_model = join(nnunet_dir, 'models')

    # Create new directory for the model
    model_name = 'default_model'
    model_dir = join(path_save_model, model_name)
    maybe_mkdir_p(model_dir)

    ################################### Training #############################################

    for fold in range(FOLDS):
        if fold < SKIP_FOLD:
            # If SKIP_FOLD is used, training on prior folds will be skipped
            print('Skipping fold', fold)
            print('                   ')
        else:
            print('Running training: fold', fold)
            print('                            ')

            start = time()

            os.system(f'nnUNet_train 3d_fullres nnUNetTrainerV2 Task100_grid {str(fold)}')

            print('                                 ')
            print('End of training: fold',       fold)
            print(f'Training took {time() - start} s')
            print('                                 ')

            # Create fold directory if necessary
            fold_dir = join(model_dir, f'fold{fold}')
            maybe_mkdir_p(fold_dir)

            # Copy files from nnunet directory to personal directory
            shutil.copyfile(join(path_models, f'fold_{fold}', '*'), model_dir + '/')
            shutil.copyfile(join(nnunet_dir, 'nnUNet_base/nnUNet_raw/Task100_grid/dataset.json'), model_dir +'/')

            print('Files from present fold transferred to', fold_dir)
            print('                                                ')

    print('End of training')



##############################################################################################
#--------------------------------------------------------------------------------------------#    
##############################################################################################



def file_management(nnunet_dir, SEED=0, DATASET_SIZE=None):
    ''' ############################# File management ########################################

    File management for training/testing of a nnunet model.

    Images (labels) should be in the database_images/ (database_labels/) 
    dir inside: ./nnunet/nnUNet_base/nnUNet_raw/Task00_grid/ 

    Git repository: https://github.com/perecanals/nnunet_vh2020.git

    '''
    ################################# File management ########################################

    # Paths
    path_images_base = join(nnunet_dir, 'database_vh/database_images')
    path_labels_base = join(nnunet_dir, 'database_vh/database_labels')

    path_imagesTr = join(nnunet_dir, 'nnUNet_base/nnUNet_raw/Task100_grid/imagesTr')
    path_labelsTr = join(nnunet_dir, 'nnUNet_base/nnUNet_raw/Task100_grid/labelsTr')
    path_imagesTs = join(nnunet_dir, 'nnUNet_base/nnUNet_raw/Task100_grid/imagesTs')

    imagesTr = './imagesTr/'
    labelsTr = './labelsTr/'
    imagesTs = './imagesTs/'

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

    print('Total number of images in the database:', len(list_images_base))
    print('                                                              ')

    # We separate a number of images for testing
    print('Dividing dataset into training and testing...')
    print('                                             ')

    if DATASET_SIZE is None:
        tr_prop   = 0.7 # includes validation (5:1)
        ts_prop = 1.0 - tr_prop
        samp_tr   = int(np.round(tr_prop * len(list_images_base)))
        samp_ts = int(np.round(ts_prop * len(list_images_base)))
        while samp_tr + samp_ts > len(list_images_base):
            samp_ts += -1
        print('Training:testing = {}:{}'.format(int(100 * tr_prop), int(100 * ts_prop)))
    else:
        samp_tr = DATASET_SIZE
        samp_ts = len(list_images_base) - samp_tr
        print('Training dataset size is manually set to', DATASET_SIZE)

    print('                                            ')
    print('Number of images used for training:', samp_tr)
    print('Number of images used for testing: ', samp_ts)
    print('                                            ')

    # We generate an order vector to shuffle the samples for training
    print(f'Shuffling dataset (SEED = {SEED})')

    np.random.seed(SEED)
    order = np.arange(len(list_images_base))
    np.random.shuffle(order)

    list_images_base_tr = [list_images_base[i] for i in order]
    list_labels_base_tr = [list_labels_base[i] for i in order]

    print('done')
    print('    ')

    # Remove all preexisting nifti files
    print('Removing possibly preexisting nifti files...')

    for files in glob.glob(join(path_imagesTr, '*.gz')):
        os.remove(files)
    for files in glob.glob(join(path_labelsTr, '*.gz')):
        os.remove(files)
    for files in glob.glob(join(path_imagesTs, '*.gz')):
        os.remove(files)
    list_imagesTr   = list_images_base_tr[:samp_tr]
    list_labelsTr   = list_labels_base_tr[:samp_tr]
    list_imagesTs = list_images_base_tr[samp_tr:]

    print('done')
    print('    ')

    # Copy files to corresponding directories
    print('Copying new files...')

    for image in list_imagesTr:
        shutil.copyfile(join(path_images_base, image), join(path_imagesTr, image))
    for label in list_labelsTr:
        shutil.copyfile(join(path_labels_base, label), join(path_labelsTr, label))
    for image in list_imagesTs:
        shutil.copyfile(join(path_images_base, image), join(path_imagesTs, image))

    print('done')
    print('    ')

    # Write the .json file 
    print('Writing .json file in preparation for training...')

    list_imagesTr_json = [None] * len(list_imagesTr)
    list_labelsTr_json = [None] * len(list_labelsTr)
    list_imagesTs_json = [None] * len(list_imagesTs)
    for idx, _ in enumerate(list_imagesTr):
        list_imagesTr_json[idx] = imagesTr + list_imagesTr[idx]
        list_labelsTr_json[idx] = labelsTr + list_labelsTr[idx]
    for idx, _ in enumerate(list_imagesTs):
        list_imagesTs_json[idx] = imagesTs + list_imagesTs[idx]

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
        "numTest": samp_ts,
        "training": [],
        "test": []
    }

    # Prepare the training and testing samples for the json file
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

    # Move json file to nnUNet_base dirs
    print('Moving .json to nnUNet_base directories...')
    os.rename(nnunet_dir + "/dataset.json", nnunet_dir + "/nnUNet_base/nnUNet_raw/Task00_grid/dataset.json")
    shutil.copyfile(nnunet_dir + "/nnUNet_base/nnUNet_raw/Task100_grid/dataset.json", nnunet_dir + "/nnUNet_base/nnUNet_preprocessed/Task100_grid/dataset.json")
    shutil.copyfile(nnunet_dir + "/nnUNet_base/nnUNet_raw/Task100_grid/dataset.json", nnunet_dir + "/nnUNet_base/nnUNet_raw_cropped/Task100_grid/dataset.json")
    shutil.copyfile(nnunet_dir + "/nnUNet_base/nnUNet_raw/Task100_grid/dataset.json", nnunet_dir + "/nnUNet_base/nnUNet_raw_splitted/Task100_grid/dataset.json")

    print('done')
    print('    ')

    print('File management finished, ready for training or testing')



##############################################################################################
#--------------------------------------------------------------------------------------------#    
##############################################################################################


