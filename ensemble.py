import numpy as np

import os
from glob import glob
import shutil
import json
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

from evaluation_metrics import eval_metrics

from time import time

def ensemble_single(nnunet_dir, INPUT_DIR, INPUT, MODEL_DIR=None, MODEL=None, LOWRES=True):

    # INPUT_DIR = "TFM/nnunet_env/nnUNet/nnunet/inference_test/input"
    # INPUT = "15308585.nii.gz"

    if MODEL_DIR is None: ValueError('Please input path to model. See main.py -h for more information.')

    pat_id = INPUT[:-7]

    input_path = os.path.join(INPUT_DIR, pat_id, 'input')
    maybe_mkdir_p(input_path)
    
    if not os.path.isfile(os.path.join(input_path, pat_id + '_0000.nii.gz')):
        os.rename(os.path.join(INPUT_DIR, INPUT), os.path.join(input_path, pat_id + '_0000.nii.gz'))

    npz_folders = []

    start = time()

    for fold in range(5):
        print(f'Predicting {pat_id} fold {fold}')
        print('                                ')

        output_path = os.path.join(INPUT_DIR, pat_id, f'fold_{fold}')
        maybe_mkdir_p(output_path)

        FOLD_DIR = os.path.join(MODEL_DIR, f'fold_{fold}')

        model = os.path.join(FOLD_DIR, 'model_best.model')
        pkl   = os.path.join(FOLD_DIR, 'model_best.model.pkl')
        plans = os.path.join(FOLD_DIR[:-7], 'plans.pkl')

        if LOWRES:
            path_models = os.path.join(nnunet_dir, f'nnUNet_base/nnUNet_training_output_dir/nnUNet/3d_lowres/Task100_grid/nnUNetTrainerV2__nnUNetPlansv2.1/fold_{fold}')
        else:
            path_models = os.path.join(nnunet_dir, f'nnUNet_base/nnUNet_training_output_dir/nnUNet/3d_fullres/Task100_grid/nnUNetTrainerV2__nnUNetPlansv2.1/fold_{fold}')

        maybe_mkdir_p(path_models)

        # Copy selected model to nnunet output dir
        print('Copying selected model to nnUNet output dir for inference...')

        shutil.copyfile(model, os.path.join(path_models, 'model_final_checkpoint.model'))
        shutil.copyfile(pkl,   os.path.join(path_models, 'model_final_checkpoint.model.pkl'))
        shutil.copyfile(plans, os.path.join(path_models[:-7], 'plans.pkl'))

        # if input_path[:8] == '/content': # We are working on drive, we need to get rid of spaces in the path
        #     input_path_aux = input_path[:17] + '\ ' + input_path[18:]
        #     output_path_aux = output_path[:17] + '\ ' + output_path[18:]

        if LOWRES:
            os.system('nnUNet_predict -i ' + input_path + ' -o ' + output_path + ' -t Task100_grid -z -m 3d_lowres -f ' + str(fold))
        else:
            os.system('nnUNet_predict -i ' + input_path + ' -o ' + output_path + ' -t Task100_grid -z store_true -m 3d_fullres -f ' + str(fold))

        npz_folders.append(output_path)

        print(f'Fold {fold} completed')
        print('                      ')

    output_folder = os.path.join(INPUT_DIR, pat_id, 'output')

    if input_path[:8] == '/content':
        output_folder = output_folder[:17] + '\ ' + output_folder[18:]

    print('Starting ensembling')

    os.system(f'nnUNet_ensemble -f {npz_folders[0]} {npz_folders[1]} {npz_folders[2]} {npz_folders[3]} {npz_folders[4]} -o {output_folder}')

    print(f'Ensemble took {time() - start} s')
    print('                                  ')