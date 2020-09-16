import os
import shutil
from glob import glob
import argparse

####################################### Arguments ############################################

parser = argparse.ArgumentParser()

parser.add_argument('-model', '--model', type=str, required=True, 
    help='Please input a model name. Required.')

parser.add_argument('-s', '--seed', type=int, required=False, default=0, 
    help='random seed for the dataset ordering (training and testing) required.')

parser.add_argument('-f', '--folds', type=int, required=False, 
    help='number of folds for the cross validation. If we do not input FOLDS in ::TRAINING:: mode, '
         'we will train with the whole training set.')

parser.add_argument('-sk', '--skip_fold', type=int, required=False, default=0, 
    help='it skips folds below number specified. E.g. if --skip_fold is set to ::1::, it will '
         'skip the first fold (fold 0). Not required.')

parser.add_argument('-lowres', '--lowres', type=bool, required=False, default=False,
    help='use in case you want to work with lowres images. Not required.')

parser.add_argument('-trainer', '--trainer', type=str, required=False, default='default',
    help='choose a trainer class from those in nnunet.training.network.training. '
         'Choose between ::initial_lr_1e3::, ::Adam::, ::initial_lr_1e3_Adam::, ::SGD_ReduceOnPlateau::, '
         '::Loss_Jacc_CE:: or ::initial_lr_1e3_Loss_Jacc_CE::. Not required.')


args = parser.parse_args()

MODEL        = args.model
SEED         = args.seed
FOLDS        = args.folds
SKIP_FOLD    = args.skip_fold
LOWRES       = args.lowres
TRAINER      = args.trainer

if LOWRES:
    MODEL = MODEL + '_lowres'

MODEL = MODEL + f'_{TRAINER}_{SEED}'

print('Transferring data from model ', MODEL)
print('                                    ')

if TRAINER == 'default':
    trainer = 'nnUNetTrainerV2'
elif TRAINER == 'initial_lr_1e3':
    trainer = 'nnUNetTrainerV2_initial_lr_1e3'
elif TRAINER == 'Adam':
    trainer = 'nnUNetTrainerV2_Adam'
elif TRAINER == 'initial_lr_1e3_Adam':
    trainer = 'nnUNetTrainerV2_initial_lr_1e3_Adam'
elif TRAINER == 'initial_lr_1e3_SGD_ReduceOnPlateau':
    trainer = 'nnUNetTrainerV2_initial_lr_1e3_SGD_ReduceOnPlateau'
elif TRAINER == 'SGD_ReduceOnPlateau':
    trainer = 'nnUNetTrainerV2_SGD_ReduceOnPlateau'
elif TRAINER == 'Loss_Jacc_CE':
    trainer = 'nnUNetTrainerV2_Loss_Jacc_CE'
elif TRAINER == 'initial_lr_1e3_Loss_Jacc_CE':
    trainer = 'nnUNetTrainerV2_initial_lr_1e3_Loss_Jacc_CE'
else:
    NotImplementedError('Please choose one of the implemented trainers')

nnunet_dir = os.path.abspath('')

nnunet_model_dir = os.path.join(nnunet_dir, 'nnUNet_base/nnUNet_training_output_dir/nnUNet/3d_lowres/Task100_grid', trainer + '__nnUNetPlansv2.1')
models_dir = os.path.join(nnunet_dir, 'models', 'Task100_' + MODEL, trainer + '__nnUNetPlansv2.1')

if os.path.isfile(os.path.join(nnunet_model_dir, 'plans.pkl')):
    shutil.copyfile(os.path.join(nnunet_model_dir, 'plans.pkl'), os.path.join(models_dir, 'plans.pkl'))

for fold in range(FOLDS):
    if fold < SKIP_FOLD:
        print('Skipping fold', fold)
        print('                   ')
    else:
        print('Transferring data from fold_', fold)
        print('                                  ')

        for files in os.listdir(os.path.join(nnunet_model_dir, f'fold_{fold}')):
            if not os.path.isdir(os.path.join(nnunet_model_dir, f'fold_{fold}', files)):
                print(f'Transferring file {files}...')
                shutil.copyfile(os.path.join(nnunet_model_dir, f'fold_{fold}', files), os.path.join(models_dir, f'fold_{fold}', files))
                print('done')

        print(f'Fold {fold} done')
        print('                 ')