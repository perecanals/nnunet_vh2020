import numpy as np

import os
from glob import glob
import shutil
import json
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

from evaluation_metrics import eval_metrics

from time import time

def inference_single(nnunet_dir, INPUT_DIR, INPUT, MODEL_DIR=None, MODEL=None, LOWRES=True, TEST_FOLD=0 ):

    if MODEL_DIR is None: ValueError('Please input path to model. See main.py -h for more information.')

    if TEST_FOLD not in [0, 1, 2, 3, 4]:
        TEST_FOLD = 'all'
        fold = 'all'
    else:
        TEST_FOLD = str(TEST_FOLD)
        fold = 'fold_' + TEST_FOLD

    pat_id = INPUT[:-7]

    input_path = os.path.join(INPUT_DIR, pat_id, 'input')
    output_path = os.path.join(INPUT_DIR, pat_id, 'output')

    maybe_mkdir_p(input_path)
    maybe_mkdir_p(output_path)
    
    if not os.path.isfile(os.path.join(input_path, pat_id + '_0000.nii.gz')):
        os.rename(os.path.join(INPUT_DIR, INPUT), os.path.join(input_path, pat_id + '_0000.nii.gz'))

    model = os.path.join(MODEL_DIR, 'model_best.model')
    pkl   = os.path.join(MODEL_DIR, 'model_best.model.pkl')
    plans = os.path.join(MODEL_DIR[:-7], 'plans.pkl')

    if LOWRES:
        path_models = os.path.join(nnunet_dir, f'nnUNet_base/nnUNet_training_output_dir/nnUNet/3d_lowres/Task100_grid/nnUNetTrainerV2__nnUNetPlansv2.1/{fold}')
    else:
        path_models = os.path.join(nnunet_dir, f'nnUNet_base/nnUNet_training_output_dir/nnUNet/3d_fullres/Task100_grid/nnUNetTrainerV2__nnUNetPlansv2.1/{fold}')

    maybe_mkdir_p(path_models)

    # Copy selected model to nnunet output dir
    print('Copying selected model to nnUNet output dir for inference...')

    shutil.copyfile(model, os.path.join(path_models, 'model_final_checkpoint.model'))
    shutil.copyfile(pkl,   os.path.join(path_models, 'model_final_checkpoint.model.pkl'))
    shutil.copyfile(plans, os.path.join(path_models[:-len(fold)], 'plans.pkl'))

    if input_path[:8] == '/content': # We are working on drive, we need to get rid of spaces in the path
        input_path = input_path[:17] + '\ ' + input_path[18:]
        output_path = output_path[:17] + '\ ' + output_path[18:]

    start = time()

    if LOWRES:
        os.system('nnUNet_predict -i ' + input_path + ' -o ' + output_path + f' -t Task100_grid -m 3d_lowres -f ' + TEST_FOLD)
    else:
        os.system('nnUNet_predict -i ' + input_path + ' -o ' + output_path + f' -t Task100_grid -m 3d_fullres -f ' + TEST_FOLD)

    print(f'Inference took {time() - start} s')
    print('                                  ')