import numpy as np

import os
from glob import glob
import shutil
import json

from time import time

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

def training(nnunet_dir, FOLDS=1, SKIP_FOLD=0, MODEL=None, LOWRES=False, CONTINUE=False, trainer='nnUNetTrainerV2'):
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

            start = time()

            if LOWRES:
                print('Using low resolution (1/3) images')
                if CONTINUE:
                    print('Continuing training')
                    # os.rename(os.path.join(nnunet_dir, f'nnUNet_base/nnUNet_training_output_dir/nnUNet/3d_lowres/Task100_{MODEL}'), os.path.join(nnunet_dir, 'nnUNet_base/nnUNet_training_output_dir/nnUNet/3d_lowres/Task100_grid'))
                    os.system('nnUNet_train 3d_lowres ' + trainer + f' Task100_grid {str(fold)} -c')
                else:
                    os.system('nnUNet_train 3d_lowres ' + trainer + f' Task100_grid {str(fold)}')
            else:
                print('Using full resolution images')
                if CONTINUE:
                    print('Continuing training')
                    # os.rename(os.path.join(nnunet_dir, f'nnUNet_base/nnUNet_training_output_dir/nnUNet/3d_lowres/Task100_{MODEL}'), os.path.join(nnunet_dir, 'nnUNet_base/nnUNet_training_output_dir/nnUNet/3d_lowres/Task100_grid'))
                    os.system('nnUNet_train 3d_fullres ' + trainer + f' Task100_grid {str(fold)} -c')
                else:
                    os.system('nnUNet_train 3d_fullres ' + trainer + f' Task100_grid {str(fold)}')

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



def file_management(nnunet_dir, SEED=0, DATASET_SIZE=None, MODEL=None, LOWRES=False, trainer='nnUNetTrainerV2'):
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
        tr_prop = 0.7 # includes validation (5:1)
        ts_prop = 1.0 - tr_prop
        samp_tr = int(np.round(tr_prop * len(list_images_base)))
        samp_ts = int(np.round(ts_prop * len(list_images_base)))
        while samp_tr + samp_ts > len(list_images_base):
            samp_ts += -1
        print('Training:testing = {}:{}'.format(int(100 * tr_prop), int(100 * ts_prop)))
    else:
        samp_tr = DATASET_SIZE
        if len(list_images_base) > 2 * DATASET_SIZE:
            samp_ts = DATASET_SIZE
        else:
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

    list_imagesTr = list_images_base_tr[:samp_tr]
    list_labelsTr = list_labels_base_tr[:samp_tr]
    list_imagesTs = list_images_base_tr[samp_tr:samp_tr+samp_ts]

    pat_ids = []
    for pat_id in list_imagesTr:
        pat_ids.append(pat_id[-15:-7])

    # Remove all preexisting nifti files
    print('Removing possibly preexisting nifti files...')

    for files in glob(os.path.join(path_imagesTr, '*.gz')):
        if (files[-20:-12] + '.nii.gz') not in list_imagesTr:
            os.remove(files)
    for files in glob(os.path.join(path_labelsTr, '*.gz')):
        if files[-15:] not in list_imagesTr:
            os.remove(files)
    for files in glob(os.path.join(path_preprocessed, '*.npz')):
        if files[-12:-4] not in pat_ids:
            os.remove(files)
    for files in glob(os.path.join(path_preprocessed, '*.npy')):
        if files[-12:-4] not in pat_ids:
            os.remove(files)
    for files in glob(os.path.join(path_preprocessed, '*.pkl')):
        if files[-12:-4] not in pat_ids:
            os.remove(files)

    print('done')
    print('    ')

    # Copy files to corresponding directories
    print('Copying new files...')
    for image in list_imagesTr:
        shutil.copyfile(os.path.join(path_images_base, image), os.path.join(path_imagesTr, image[:8] + '_0000.nii.gz'))
    for label in list_labelsTr:
        shutil.copyfile(os.path.join(path_labels_base, label), os.path.join(path_labelsTr, label))
        shutil.copyfile(os.path.join(path_preprocessed_all, label[:8] + '.npz'), os.path.join(path_preprocessed, label[:8] + '.npz'))
        shutil.copyfile(os.path.join(path_preprocessed_all, label[:8] + '.npy'), os.path.join(path_preprocessed, label[:8] + '.npy'))
        shutil.copyfile(os.path.join(path_preprocessed_all, label[:8] + '.pkl'), os.path.join(path_preprocessed, label[:8] + '.pkl'))


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

    list_imagesTr_json = sorted(list_imagesTr_json)
    list_labelsTr_json = sorted(list_labelsTr_json)
    list_imagesTs_json = sorted(list_imagesTs_json)   

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

    aux2 = []
    for idx, _ in enumerate(list_imagesTs_json):
        aux2 = np.append(aux2, {
                        "image": list_imagesTs_json[idx],
                    })

    dataset['training'] = list(aux)
    dataset['test'] = list(aux2)

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


