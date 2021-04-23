import numpy as np

import os
from glob import glob
import shutil
import json
import pickle as pkl

from time import time

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

def training(nnunet_dir, FOLDS=1, SKIP_FOLD=0, MODEL=None, LOWRES=False, CONTINUE=False, trainer='nnUNetTrainerV2', DATASET_CONFIG=1):
    ''' ############################### Training #############################################

    Training of the nnunet (3-fold cross validation).

    We perform training for each of the modifications that we add
    to the nnUNet framewrok. This includes (0) default out-of-the-box 
    nnunet, (1) changes in data augmentation, (2) lower resolution 
    of the dataset.

    Preprocessing of the whole dataset and file management should be 
    perform prior to this script.

    Images (labels) should be in the database_images/ (database_labels/) 
    dir inside: ./nnunet/nnUNet_base/nnUNet_raw/Task100_grid/ 

    Git repository: https://github.com/perecanals/nnunet_vh2020.git
    Original nnunet (Isensee et al. 2020[1]): https://github.com/MIC-DKFZ/nnUNet.git

    [1] Fabian Isensee, Paul F. Jäger, Simon A. A. Kohl, Jens Petersen, Klaus H. Maier-Hein "Automated Design of Deep Learning 
    Methods for Biomedical Image Segmentation" arXiv preprint arXiv:1904.08128 (2020).

    '''

    if trainer == 'nnUNetTrainerV2':
        print('Using default trainer')
        print('                     ')
    elif trainer == 'nnUNetTrainerV2_initial_lr_1e3':
        print('Starting with an learning rate of 1e-3')
        print('                                      ')
    elif trainer == 'nnUNetTrainerV2_Adam':
        print('Using Adam as optimizer')
        print('                       ')
    elif trainer == 'nnUNetTrainerV2_initial_lr_1e3_Adam':
        print('Starting with an learning rate of 1e-3 and Adam')
        print('                                      ')
    elif trainer == 'nnUNetTrainerV2_SGD_ReduceOnPlateau':
        print('Using SGD_ReduceLROnPlateau')
        print('                       ')
    elif trainer == 'initial_lr_1e3_SGD_ReduceOnPlateau':
        print('Using initial_lr_1e3_SGD_ReduceOnPlateau')
        print('                                        ')
    elif trainer == 'nnUNetTrainerV2_Loss_Jacc_CE':
        print('Using Jaccard and CE loss')
        print('                         ')
    elif trainer == 'nnUNetTrainerV2_initial_lr_1e3_Loss_Jacc_CE':
        print('Using Jaccard and CE loss and initial lr of 1e-3')
        print('                                                ')


    # Paths
    if LOWRES:
        path_models = os.path.join(nnunet_dir, 'nnUNet_base/nnUNet_training_output_dir/nnUNet/3d_lowres/Task100_grid/' + trainer + '__nnUNetPlansv2.1')
    else:
        path_models = os.path.join(nnunet_dir, 'nnUNet_base/nnUNet_training_output_dir/nnUNet/3d_fullres/Task100_grid/' + trainer + '__nnUNetPlansv2.1')
    
    path_preprocessed = os.path.join(nnunet_dir, 'nnUNet_base/nnUNet_preprocessed/Task100_grid')

    # Create new directory for the model
    model_dir = os.path.join(nnunet_dir, 'models', 'Task100_' + MODEL, trainer + '__nnUNetPlansv2.1')
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

            # Create fold directory if necessary
            fold_dir = os.path.join(model_dir, f'fold_{fold}')
            maybe_mkdir_p(fold_dir)

            if DATASET_CONFIG in [0, 1] and not CONTINUE: # If CONTINUE = True, then splits_final.pkl should be in `fold_dir`
                generate_splits(nnunet_dir, fold, model_dir, DATASET_CONFIG)
            elif DATASET_CONFIG in [0, 1] and CONTINUE:
                shutil.copyfile(os.path.join(fold_dir, 'splits_final.pkl'), os.path.join(nnunet_dir, 'nnUNet_base/nnUNet_preprocessed/Task100_grid', 'splits_final.pkl'))

            start = time()

            fold_aux = None
            if fold in [3, 4]: # For some reason, folds 3 and 4 often get "Killed" in Colab. Only use this if you know what you are doing. You shoud copy
                               # somewhere else all data from folds 0 and 1 before training folds 3 and 4. Otherwise it will be a mess
                path = os.path.join(nnunet_dir, 'nnUNet_base/nnUNet_preprocessed/Task100_grid', 'splits_final.pkl')
                with open(path, 'rb') as handle:
                    data = pkl.load(handle)

                data[0] = data[3]
                data[1] = data[4]

                with open(path, 'wb') as handle:
                    pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)

                fold_aux = fold
                if fold == 3: 
                    fold = 0
                if fold == 4: 
                    fold = 1

            if LOWRES:
                print('Using low resolution (1/3) images')
                if CONTINUE:
                    print('Continuing training')
                    if DATASET_CONFIG in [1, 2]: shutil.copyfile(os.path.join(fold_dir, 'splits_final.pkl'), os.path.join(path_preprocessed, 'splits_final.pkl'))
                    os.system('nnUNet_train 3d_lowres ' + trainer + f' Task100_grid {str(fold)} -c')
                else:
                    os.system('nnUNet_train 3d_lowres ' + trainer + f' Task100_grid {str(fold)}')
            else:
                print('Using full resolution images')
                if CONTINUE:
                    print('Continuing training')
                    if DATASET_CONFIG in [1, 2]: shutil.copyfile(os.path.join(fold_dir, 'splits_final.pkl'), os.path.join(path_preprocessed, 'splits_final.pkl'))
                    os.system('nnUNet_train 3d_fullres ' + trainer + f' Task100_grid {str(fold)} -c')
                else:
                    os.system('nnUNet_train 3d_fullres ' + trainer + f' Task100_grid {str(fold)}')

            if fold_aux is not None: fold = fold_aux

            print('                                 ')
            print('End of training: fold',       fold)
            print(f'Training took {time() - start} s')
            print('                                 ')

            # Copy files from nnunet directory to personal directory
            for files in os.listdir(os.path.join(path_models, f'fold_{fold}')):
                shutil.copyfile(os.path.join(path_models, f'fold_{fold}', files), os.path.join(fold_dir, files))

            # Renaming directory to identify better
            # os.rename(os.path.join(nnunet_dir, 'nnUNet_base/nnUNet_training_output_dir/nnUNet/3d_lowres/Task100_grid'), os.path.join(nnunet_dir, f'nnUNet_base/nnUNet_training_output_dir/nnUNet/3d_lowres/Task100_{MODEL}'))

            print('Files from present fold transferred to', fold_dir)
            print('                                                ')

    print('End of training')



##############################################################################################
#--------------------------------------------------------------------------------------------#    
##############################################################################################



def file_management(nnunet_dir, SEED=0, DATASET_SIZE=None, MODEL=None, LOWRES=False, trainer='nnUNetTrainerV2', DATASET_CONFIG=0):
    ''' ############################# File management ########################################

    File management for training/testing of a nnunet model.

    Images (labels) should be in the database_images/ (database_labels/) 
    dir inside: ./nnunet/nnUNet_base/nnUNet_raw/Task00_grid/ 

    Git repository: https://github.com/perecanals/nnunet_vh2020.git

    '''
    ################################# File management ########################################

    if MODEL is None:
        raise ValueError('Please enter a valid model name')

    # Paths
    path_images_base = os.path.join(nnunet_dir, 'database_vh/database_images')
    path_labels_base = os.path.join(nnunet_dir, 'database_vh/database_labels')

    path_imagesTr = os.path.join(nnunet_dir, 'nnUNet_base/nnUNet_raw_data/Task100_grid/imagesTr')
    path_labelsTr = os.path.join(nnunet_dir, 'nnUNet_base/nnUNet_raw_data/Task100_grid/labelsTr')
    path_imagesTs = os.path.join(nnunet_dir, 'nnUNet_base/nnUNet_raw_data/Task100_grid/imagesTs')

    path_preprocessed = os.path.join(nnunet_dir, 'nnUNet_base/nnUNet_preprocessed/Task100_grid/nnUNetData_plans_v2.1_stage0')
    path_preprocessed_all = os.path.join(nnunet_dir, 'nnUNet_base/nnUNet_preprocessed/Task100_grid/nnUNetData_plans_v2.1_stage0/all')

    if os.path.exists(os.path.join(nnunet_dir, 'nnUNet_base/nnUNet_preprocessed/Task100_grid/splits_final.pkl')):
        os.remove(os.path.join(nnunet_dir, 'nnUNet_base/nnUNet_preprocessed/Task100_grid/splits_final.pkl'))

    imagesTr = './imagesTr/'
    labelsTr = './labelsTr/'
    imagesTs = './imagesTs/'

    # List all available images and labels
    list_images_base = sorted([filename for filename in os.listdir(path_images_base) if filename.endswith('.nii.gz')])
    list_labels_base = sorted([filename for filename in os.listdir(path_labels_base) if filename.endswith('.nii.gz')])

    print('Total number of images in the database:', len(list_images_base))
    print('                                                              ')

    # We separate a number of images for testing
    print('Dividing dataset into training and testing...')
    print('                                             ')

    if DATASET_SIZE is None:
        tr_prop = 0.8 # includes validation (5:1)
        ts_prop = 1.0 - tr_prop
        samp_tr = int(np.round(tr_prop * len(list_images_base)))
        samp_ts = int(np.round(ts_prop * len(list_images_base)))
        while samp_tr + samp_ts > len(list_images_base):
            samp_ts += -1
        print('Training:testing = {}:{}'.format(int(100 * tr_prop), int(100 * ts_prop)))
    elif DATASET_SIZE is not None and DATASET_CONFIG == 0:
        samp_tr = DATASET_SIZE
        if len(list_images_base) > 2 * DATASET_SIZE: # Set a maximum for the testing set size in case DATASET_SIZE is small
            samp_ts = DATASET_SIZE
        else:
            samp_ts = len(list_images_base) - samp_tr
        print('Training dataset size is manually set to', DATASET_SIZE)
    elif DATASET_SIZE is not None and DATASET_CONFIG == 1:
        tr_prop = 0.8 # includes validation (5:1)
        ts_prop = 1.0 - tr_prop
        samp_tr = int(np.round(tr_prop * DATASET_SIZE))
        samp_ts = int(np.round(ts_prop * DATASET_SIZE))
        while samp_tr + samp_ts > DATASET_SIZE:
            samp_ts += -1
        print('Training:testing = {}:{}'.format(samp_tr, samp_ts))

    print('                                            ')
    print('Number of images used for training:', samp_tr)
    print('Number of images used for testing: ', samp_ts)
    print('                                            ')

    # We generate an order vector to shuffle the samples for training
    print(f'Shuffling dataset (SEED = {SEED})')

    np.random.seed(SEED)
    order = np.arange(len(list_images_base))
    np.random.shuffle(order)

    list_images_base_sh = [list_images_base[i] for i in order]
    list_labels_base_sh = [list_labels_base[i] for i in order]
    
    print('done')
    print('    ')
        
    # Use DATASET_CONFIG 0 if you want to use the same testing set for each fold, using the rest of the dataset for training+validation only
    # Here, DATASET_SIZE indicates the size of the training+validation set
    # Not choosing a DATASET_SIZE will mean to select a random testing set (with a 80:20 ratio)
    if DATASET_CONFIG == 0:

        dataset = {}
        dataset = {
            "name": "StrokeVessels",
            "description": "Upper Trunk Vessels Segmentation",
            "reference": "Hospital Vall dHebron",
            "licence": "-",
            "release": "1.0 08/01/2020",
            "tensorImageSize": "3D",
            "lowres": LOWRES,
            "trainer": trainer,
            "modality": {
                "0": "CT"
            },
            "labels": {
                "0": "background",
                "1": "vessel"
            },
            "model": MODEL,
            "dataset config": DATASET_CONFIG,
            "dataset size": DATASET_SIZE,
            "numTraining": samp_tr,
            "numTest": samp_ts,
            "fold 0": {
                "training": [],
            },
            "fold 1": {
                "training": [],
            },
            "fold 2": {
                "training": [],
            },
            "fold 3": {
                "training": [],
            },
            "fold 4": {
                "training": [],
            }
        }

        if DATASET_SIZE is not None:
            list_images_base_tr = list_images_base_sh[:DATASET_SIZE]
            list_images_base_ts = list_images_base_sh[DATASET_SIZE:]

        array_images_base_tr = np.array(list_images_base_tr)
        array_images_base_ts = np.array(list_images_base_ts)

        folds = 5 # Set to 5 folds

        for fold in range(folds):
            imagesTr_fold = array_images_base_tr
            imagesTs_fold = array_images_base_ts

            imagesTr_fold_json = [None] * len(imagesTr_fold)
            labelsTr_fold_json = [None] * len(imagesTr_fold)
            imagesTs_fold_json = [None] * len(imagesTs_fold)
            for idx, _ in enumerate(imagesTr_fold):
                imagesTr_fold_json[idx] = imagesTr + imagesTr_fold[idx]
                labelsTr_fold_json[idx] = labelsTr + imagesTr_fold[idx]
            for idx, _ in enumerate(imagesTs_fold):
                imagesTs_fold_json[idx] = imagesTs + imagesTs_fold[idx]

            imagesTr_fold_json = sorted(imagesTr_fold_json)
            labelsTr_fold_json = sorted(labelsTr_fold_json)
            imagesTs_fold_json = sorted(imagesTs_fold_json)   

            # Prepare the training and testing samples for the json file
            aux = []
            for idx, _ in enumerate(imagesTr_fold_json):
                aux = np.append(aux, {
                                "image": imagesTr_fold_json[idx],
                                "label": labelsTr_fold_json[idx]
                            })

            aux2 = []
            for idx, _ in enumerate(imagesTs_fold_json):
                aux2 = np.append(aux2, {
                                "image": imagesTs_fold_json[idx],
                            })

            dataset[f'fold {fold}']['training'] = list(aux)
            dataset[f'fold {fold}']['test'] = list(aux2)
    
    # Use DATASET_CONFIG 1 if you want to limit the size of the whole dataset
    # Here, DATASET_SIZE indicates the size of the training+validation+test set
    if DATASET_CONFIG == 1:

        dataset = {}
        dataset = {
            "name": "StrokeVessels",
            "description": "Upper Trunk Vessels Segmentation",
            "reference": "Hospital Vall dHebron",
            "licence": "-",
            "release": "1.0 08/01/2020",
            "tensorImageSize": "3D",
            "lowres": LOWRES,
            "trainer": trainer,
            "modality": {
                "0": "CT"
            },
            "labels": {
                "0": "background",
                "1": "vessel"
            },
            "model": MODEL,
            "dataset config": DATASET_CONFIG,
            "dataset size": DATASET_SIZE,
            "numTraining": samp_tr,
            "numTest": samp_ts,
            "fold 0": {
                "training": [],
            },
            "fold 1": {
                "training": [],
            },
            "fold 2": {
                "training": [],
            },
            "fold 3": {
                "training": [],
            },
            "fold 4": {
                "training": [],
            }
        }

        if DATASET_SIZE is not None:
            list_images_base_sh = list_images_base_sh[:DATASET_SIZE]

        array_images_base_sh = np.array(list_images_base_sh)

        folds = 5 # Set to 5 folds

        for fold in range(folds):
            if fold == 0:    
                imagesTs_fold = array_images_base_sh[: samp_ts]
                imagesTr_fold = array_images_base_sh[samp_ts:]
            elif fold == (folds - 1):
                imagesTs_fold = array_images_base_sh[fold * samp_ts:]
                imagesTr_fold = array_images_base_sh[: fold * samp_ts]
            else:
                imagesTs_fold = array_images_base_sh[fold * samp_ts: (fold + 1) * samp_ts]
                imagesTr_fold = np.append(array_images_base_sh[: fold * samp_ts], array_images_base_sh[(fold + 1) * samp_ts:])

            imagesTr_fold_json = [None] * len(imagesTr_fold)
            labelsTr_fold_json = [None] * len(imagesTr_fold)
            imagesTs_fold_json = [None] * len(imagesTs_fold)
            for idx, _ in enumerate(imagesTr_fold):
                imagesTr_fold_json[idx] = imagesTr + imagesTr_fold[idx]
                labelsTr_fold_json[idx] = labelsTr + imagesTr_fold[idx]
            for idx, _ in enumerate(imagesTs_fold):
                imagesTs_fold_json[idx] = imagesTs + imagesTs_fold[idx]

            imagesTr_fold_json = sorted(imagesTr_fold_json)
            labelsTr_fold_json = sorted(labelsTr_fold_json)
            imagesTs_fold_json = sorted(imagesTs_fold_json)   

            # Prepare the training and testing samples for the json file
            aux = []
            for idx, _ in enumerate(imagesTr_fold_json):
                aux = np.append(aux, {
                                "image": imagesTr_fold_json[idx],
                                "label": labelsTr_fold_json[idx]
                            })

            aux2 = []
            for idx, _ in enumerate(imagesTs_fold_json):
                aux2 = np.append(aux2, {
                                "image": imagesTs_fold_json[idx],
                            })

            dataset[f'fold {fold}']['training'] = list(aux)
            dataset[f'fold {fold}']['test'] = list(aux2)

    with open('dataset.json', 'w') as outfile:
        json.dump(dataset, outfile, indent=4)

    print('done')
    print('    ')

    # Move json file to nnUNet_base dirs
    print('Moving .json to nnUNet_base directories...')
    os.rename(nnunet_dir + '/dataset.json', nnunet_dir + '/nnUNet_base/nnUNet_raw_data/Task100_grid/dataset.json')
    shutil.copyfile(nnunet_dir + '/nnUNet_base/nnUNet_raw_data/Task100_grid/dataset.json', nnunet_dir + '/nnUNet_base/nnUNet_preprocessed/Task100_grid/dataset.json')

    # Create new directory for the model
    model_dir = os.path.join(nnunet_dir, 'models', 'Task100_' + MODEL, trainer + '__nnUNetPlansv2.1')
    maybe_mkdir_p(model_dir)
    shutil.copyfile(os.path.join(nnunet_dir, 'nnUNet_base/nnUNet_preprocessed/Task100_grid/dataset.json'), os.path.join(model_dir, 'dataset.json'))

    print('done')
    print('    ')

    print('File management finished, ready for training or testing')



##############################################################################################
#--------------------------------------------------------------------------------------------#    
##############################################################################################



from collections import OrderedDict
import pickle

def generate_splits(nnunet_dir, FOLD, model_dir, DATASET_CONFIG):
    ''' ############################# Set fold files #########################################

    Set training and validation splits according to dataset configuration 1 (variable testing set).

    Git repository: https://github.com/perecanals/nnunet_vh2020.git

    '''
    #Paths
    path_preprocessed = os.path.join(nnunet_dir, 'nnUNet_base/nnUNet_preprocessed/Task100_grid')

    if os.path.exists(os.path.join(path_preprocessed, 'splits_final.pkl')):
        os.remove(os.path.join(path_preprocessed, 'splits_final.pkl'))

    dataset_json = os.path.join(model_dir, 'dataset.json')

    list_imagesTr = []

    with open(dataset_json) as json_file:
        data = json.load(json_file)
        assert data['dataset config'] == DATASET_CONFIG
        for image in data[f'fold {FOLD}']['training']:
            list_imagesTr.append(image['image'][-15:])

    array_imagesTr = np.array(list_imagesTr).astype('<U8')
    np.random.seed(10) # 10 set at for no reason. You could set a second random seed here
    np.random.shuffle(array_imagesTr)
    splits = []

    tr_size = round(len(array_imagesTr) * 5 / 6)
    vl_size = round(len(array_imagesTr) / 6)
    while tr_size + vl_size > len(array_imagesTr): vl_size += -1

    print('Generating splits')
    print('                 ')

    for fold in range(5):
        split_fold = OrderedDict([
            ('train', array_imagesTr[:tr_size]), 
            ('val', array_imagesTr[tr_size:])
        ])
        splits.append(split_fold)

        array_imagesTr = np.roll(array_imagesTr, vl_size)
        
        if fold == FOLD:
            print('Fold', FOLD)
            print('train:', split_fold['train'])
            print('val:', split_fold['val'])

    with open(os.path.join(path_preprocessed, 'splits_final.pkl'), 'wb') as handle:
        pickle.dump(splits, handle, protocol=pickle.HIGHEST_PROTOCOL)

    shutil.copyfile(os.path.join(path_preprocessed, 'splits_final.pkl'), os.path.join(model_dir, f'fold_{FOLD}', 'splits_final.pkl'))