import numpy as np

import os
import glob
import shutil
import json
import argparse

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

from preprocessing import preprocessing
from training import training, file_management
from testing import testing
from inference_single import inference_single
from ensemble import ensemble_single

####################################### Arguments ############################################

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--mode', type=str, required=True, 
    help='choose from the following options: ::PREPROCESSING::, ::TRAINING::, '
     '::TEST::, ::TRAIN_TEST::, ::EVAL::, ::TRAIN_EVAL::, ::ENSEMBLE_TEST::, ::ENSEMBLE_EVAL::, ::ENSEMBLE_TRAIN_TEST::, '
     '::ENSEMBLE_TRAIN_EVAL::, ::INFERENCE:: or ::ENSEMBLE::. Required.')

parser.add_argument('-model', '--model', type=str, required=True, 
    help='Please input a model name. Required.')

parser.add_argument('-s', '--seed', type=int, required=False, default=0, 
    help='random seed for the dataset ordering (training and testing) required.')

parser.add_argument('-ds', '--dataset_size', type=int, required=False, 
    help='use in case you want to limit the number of training images to a certain number')

parser.add_argument('-f', '--folds', type=int, required=False, 
    help='number of folds for the cross validation. If we do not input FOLDS in ::TRAINING:: mode, '
         'we will train with the whole training set.')

parser.add_argument('-sk', '--skip_fold', type=int, required=False, default=0, 
    help='it skips folds below number specified. E.g. if --skip_fold is set to ::1::, it will '
         'skip the first fold (fold 0). Not required.')

parser.add_argument('-dc', '--dataset_config', type=int, required=False, default=1,
    help='choose between a fixed testing set (0) or a varying testing set across folds (1).')

parser.add_argument('-sk_fm', '--skip_file_management', type=bool, required=False, default=False, 
    help='skips file management for ::TRAINING:: mode. Use in case you are sure files in imagesTr/labelsTr/imagesTs '
         'are the ones desired (e.g. working with several folds in colab). Not required.')

parser.add_argument('-ram', '--low_ram', type=bool, required=False, default=False,
    help='use in case of low RAM memory for ::TEST::. It will inference one image at the time. Input '
         '::y::. Not required')

parser.add_argument('-lowres', '--lowres', type=bool, required=False, default=False,
    help='use in case you want to work with lowres images. Not required.')

parser.add_argument('-c', '--cont', type=bool, required=False, default=False,
    help='use in case you want to continue training. Not required.')

parser.add_argument('-trainer', '--trainer', type=str, required=False, default='default',
    help='choose a trainer class from those in nnunet.training.network.training. '
         'Choose between ::initial_lr_1e3::, ::Adam::, ::initial_lr_1e3_Adam::, ::SGD_ReduceOnPlateau::, '
         '::Loss_Jacc_CE:: or ::initial_lr_1e3_Loss_Jacc_CE::. Not required.')

parser.add_argument('-tf', '--test_fold', type=int, required=False, 
    help='number of fold directory that will be used for testing. If we do not input TEST_FOLD in ::TESTING:: mode, '
         'we will be using fold 0 directory for testing: '
         'os.path.join(os.path.abspath(''), nnUNet_base/nnUNet_training_output_dir/nnUNet/3d_lowres/Task100_grid/nnUNetTrainerV2__nnUNetPlansv2.1/fold_::TEST_FOLD::)')

parser.add_argument('-input_dir', '--input_dir', type=str, required=False,
    help='path to directory of input image in ::INFERENCE:: mode. Required in ::INFERENCE:: mode.')

parser.add_argument('-input', '--input', type=str, required=False,
    help='identifier of input image (nifty) in ::INFERENCE:: mode. Required in ::INFERENCE:: mode.')


args = parser.parse_args()

MODE           = args.mode
MODEL          = args.model
SEED           = args.seed
DATASET_SIZE   = args.dataset_size
FOLDS          = args.folds
SKIP_FOLD      = args.skip_fold
DATASET_CONFIG = args.dataset_config
SKIP_FM        = args.skip_file_management
LOW_RAM        = args.low_ram
LOWRES         = args.lowres
CONTINUE       = args.cont
TRAINER        = args.trainer
TEST_FOLD      = args.test_fold
INPUT_DIR      = args.input_dir
INPUT          = args.input

if MODE != 'PREPROCESSING':
    if LOWRES:
        MODEL = MODEL + '_lowres'

    MODEL = MODEL + f'_{TRAINER}_{SEED}'

    if TRAINER == 'default':
        trainer = 'nnUNetTrainerV2'
    elif TRAINER == 'initial_lr_1e3':
        trainer = 'nnUNetTrainerV2_initial_lr_1e3'
    elif TRAINER == 'Adam':
        trainer = 'nnUNetTrainerV2_Adam'
    elif TRAINER == 'initial_lr_1e3_Adam':
        trainer = 'nnUNetTrainerV2_initial_lr_1e3_Adam'
    elif TRAINER == 'SGD_ReduceOnPlateau':
        trainer = 'nnUNetTrainerV2_SGD_ReduceOnPlateau'
    elif TRAINER == 'initial_lr_1e3_SGD_ReduceOnPlateau':
        trainer = 'nnUNetTrainerV2_initial_lr_1e3_SGD_ReduceOnPlateau'
    elif TRAINER == 'Loss_Jacc_CE':
        trainer = 'nnUNetTrainerV2_Loss_Jacc_CE'
    elif TRAINER == 'initial_lr_1e3_Loss_Jacc_CE':
        trainer = 'nnUNetTrainerV2_initial_lr_1e3_Loss_Jacc_CE'
    else:
        NotImplementedError('Please choose one of the implemented trainers')
    
##############################################################################################
#--------------------------------------------------------------------------------------------#    

print('                                                                                     ')
print('Model optimization for nnUNet (Isensee et al. 2019) for segmentation of 3D CTA images')
print('By Pere Canals (2020)                                                                ')
print('                                                                                     ')

#--------------------------------------------------------------------------------------------#    
##############################################################################################

print('Working on model', MODEL)
print('                       ')

# Current directory (should be /nnunet)
nnunet_dir = os.path.abspath('')

# Set up environmental variables
os.environ['nnUNet_raw_data_base'] = os.path.join(nnunet_dir, 'nnUNet_base')
os.environ['nnUNet_preprocessed' ] = os.path.join(nnunet_dir, 'nnUNet_base', 'nnUNet_preprocessed')
os.environ['RESULTS_FOLDER'      ] = os.path.join(nnunet_dir, 'nnUNet_base', 'nnUNet_training_output_dir')

if MODE == 'PREPROCESSING':
    print('Running preprocessing')
    print('                     ')
    preprocessing(nnunet_dir)

elif MODE == 'TRAINING':
    if not SKIP_FM:
        print('Running file management')
        print('                       ')
        file_management(nnunet_dir, SEED=SEED, DATASET_SIZE=DATASET_SIZE, MODEL=MODEL, LOWRES=LOWRES, trainer=trainer, DATASET_CONFIG=DATASET_CONFIG) 

    print('Running training')
    print('                ')
    training(nnunet_dir, FOLDS=FOLDS, SKIP_FOLD=SKIP_FOLD, MODEL=MODEL, LOWRES=LOWRES, CONTINUE=CONTINUE, trainer=trainer, DATASET_CONFIG=DATASET_CONFIG)

elif MODE in ['TEST', 'EVAL', 'TRAIN_TEST', 'TRAIN_EVAL', 'ENSEMBLE_TEST', 'ENSMEBLE_EVAL', 'ENSEMBLE_TRAIN_TEST', 'ENSEMBLE_TRAIN_EVAL']:
    if TRAINER == 'default':
        trainer = 'nnUNetTrainerV2'
    elif TRAINER == 'initial_lr_1e3':
        trainer = 'nnUNetTrainerV2_initial_lr_1e3'

    # Ensembling can only be used in models trained with a fixed testing set across folds
    if MODE in ['ENSEMBLE_TEST', 'ENSMEBLE_EVAL', 'ENSEMBLE_TRAIN_TEST', 'ENSEMBLE_TRAIN_EVAL']:
        MODEL_DIR = os.path.join(nnunet_dir, 'models', 'Task100_' + MODEL, trainer + '__nnUNetPlansv2.1')
        with open(os.path.join(MODEL_DIR, 'dataset.json')) as json_file:
            data = json.load(json_file)
        assert data['dataset config'] == 0
    else:
        MODEL_DIR = os.path.join(nnunet_dir, 'models', 'Task100_' + MODEL, trainer + '__nnUNetPlansv2.1', f'fold_{FOLDS}')

    print('Running testing on model', MODEL_DIR)
    print('Mode:', MODE                        )
    print('                                   ')
    testing(nnunet_dir, MODEL_DIR=MODEL_DIR, MODE=MODE, LOW_RAM=LOW_RAM, MODEL=MODEL, FOLD=FOLDS, LOWRES=LOWRES, trainer=trainer, CONTINUE=CONTINUE, TEST_FOLD=TEST_FOLD, DATASET_CONFIG=DATASET_CONFIG)

elif MODE == 'INFERENCE':
    MODEL_DIR = os.path.join(nnunet_dir, 'models', 'Task100_' + MODEL, trainer + '__nnUNetPlansv2.1', f'fold_{FOLDS}')

    print('Performing inference on image', INPUT)
    print('Model:', MODEL_DIR                   )
    print('                                    ')
    inference_single(nnunet_dir, INPUT_DIR, INPUT, MODEL_DIR=MODEL_DIR, MODEL=MODEL, LOWRES=LOWRES, TEST_FOLD=TEST_FOLD)

elif MODE == 'ENSEMBLE':
    MODEL_DIR = os.path.join(nnunet_dir, 'models', 'Task100_' + MODEL, trainer + '__nnUNetPlansv2.1')

    print('Performing inference (ensembling) on image', INPUT)
    print('Model:', MODEL_DIR                                )
    print('                                                 ')
    ensemble_single(nnunet_dir, INPUT_DIR, INPUT, MODEL_DIR=MODEL_DIR, MODEL=MODEL, LOWRES=LOWRES)

else:
    ValueError('Please introduce one of the possible modes. See main.py -h for more information.')