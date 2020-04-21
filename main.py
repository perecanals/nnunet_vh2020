import numpy as np

import os
import glob
import shutil
import json
import argparse

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

from preprocessing import preprocessing
from training import training, file_management
from testing import testing

####################################### Arguments ############################################

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--mode', type=str, help='choose from the following options: '
                                                   '::PREPROCESSING::, ::TRAINING::, '
                                                   '::TESTING::, ::EVALUATION:: or '
                                                   '::INFERENCE::. Required.', required=True)

parser.add_argument('-s', '--seed', type=int, help='random seed for the dataset ordering. Not '
                                                   'required.', required=False)

parser.add_argument('-ds', '--dataset_size', type=int, help='use in case you want to limit the number '
                                                           'of training images to a certain number', required=False)

parser.add_argument('-f', '--folds', type=int, help='number of folds for the cross validation. '
                                                    'If we do not input FOLDS in TRAINING mode, '
                                                    'we will train with the whole training set.', required=False)

parser.add_argument('-sk', '--skip_fold', type=int, help='it skips folds below number specified. '
                                                         'E.g. if --skip_fold is set to ::1::, it will '
                                                         'skip the first fold (fold 0). Not required.', required=False)

parser.add_argument('-md', '--model_dir', type=str, help='path to the directory containing the model '
                                                         'that we want to use for TESTING. required for '
                                                         'TESTING and EVALUATION modes.', required=False)

parser.add_argument('-ram', '--low_ram', type=str, help='use in case of low RAM memory for TESTING. '
                                                        'It will inference one image at the time. Input '
                                                        '::y::. Not required', required=False)


args = parser.parse_args()

MODE           = args.mode
SEED           = args.seed
DATASET_SIZE   = args.dataset_size
FOLDS          = args.folds
SKIP_FOLD      = args.skip_fold
MODEL_DIR      = args.model_dir
LOW_RAM        = args.low_ram

if SEED is None: SEED = 0

##############################################################################################
#--------------------------------------------------------------------------------------------#    

print('                                                                                     ')
print('Model optimization for nnUNet (Isensee et al. 2019) for segmentation of 3D CTA images')
print('By Pere Canals (2020)                                                                ')
print('                                                                                     ')

#--------------------------------------------------------------------------------------------#    
##############################################################################################

# Current directory (should be /nnunet)
nnunet_dir = os.path.abspath('')

if MODE == 'PREPROCESSING':
    print('Running preprocessing')
    print('                     ')
    preprocessing(nnunet_dir)

elif MODE == 'TRAINING':
    print('Running file management')
    print('                       ')
    file_management(nnunet_dir, SEED=SEED, DATASET_SIZE=DATASET_SIZE) 

    print('Running training')
    print('                ')
    training(nnunet_dir, FOLDS=FOLDS, SKIP_FOLD=SKIP_FOLD)

elif MODE == 'TESTING' or MODE == 'EVALUATION' or MODE == 'INFERENCE':
    print('Running testing on model', MODEL_DIR)
    print('                                   ')
    testing(nnunet_dir, MODEL_DIR=MODEL_DIR, MODE=MODE, LOW_RAM=LOW_RAM)

else:
    ValueError('Please introduce one of the possible modes. See main.py -h for more information.')