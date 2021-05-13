import numpy as np

import os
from glob import glob
import shutil
import json
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

from evaluation_metrics import eval_metrics
from ensemble import ensemble_single

from time import time

def testing(nnunet_dir, MODEL_DIR=None, MODE=None, LOW_RAM=False, MODEL=None, FOLD=None, LOWRES=False, trainer='nnUNetTrainerV2', CONTINUE=False, TEST_FOLD=0, DATASET_CONFIG=1):
    ''' ################################ Testing #############################################

    Testing of the nnunet trained models. All images and labels in the 
    imagesTs section in the dataset.json in the model's directory
    should be in the database directories. The dataset.json file should
    be generated in advance, when performing training, and stored in 
    MODEL_DIR.

    Evaluation metrics computed: 
        - Dice coefficient
        - Jaccard index
        - Precision
        - Recall
        
    See evaluation_metrics.py for more information.

    Git repository: https://github.com/perecanals/nnunet_vh2020.git
    Original nnunet (Isensee et al. 2020[1]): https://github.com/MIC-DKFZ/nnUNet.git

    [1] Fabian Isensee, Paul F. Jäger, Simon A. A. Kohl, Jens Petersen, Klaus H. Maier-Hein "Automated Design of Deep Learning 
    Methods for Biomedical Image Segmentation" arXiv preprint arXiv:1904.08128 (2020).

    '''

    if MODEL_DIR is None: ValueError('Please input path to model. See main.py -h for more information.')
    if FOLD is None: ValueError('Please input fold. See main.py -h for more information.')

    if TEST_FOLD not in [0, 1, 2, 3, 4]:
        TEST_FOLD = 'all'
        fold = 'all'
    else:
        TEST_FOLD = str(TEST_FOLD)
        fold = 'fold_' + TEST_FOLD

    testing_done = False

    if MODE == 'TEST' or MODE == 'TRAIN_TEST' or MODE == 'ENSEMBLE_TEST' or MODE  == 'ENSEMBLE_TRAIN_TEST':

        ################################## Testing ###########################################

        # Paths
        path_images_base = os.path.join(nnunet_dir, 'database_vh/database_images')
        path_labels_base = os.path.join(nnunet_dir, 'database_vh/database_labels')

        maybe_mkdir_p(os.path.join(nnunet_dir, 'inference_test'))

        path_imagesTest = os.path.join(nnunet_dir, 'inference_test/input'                                   )
        path_outputTest = os.path.join(nnunet_dir, 'inference_test/output_' + trainer + '_' + MODEL_DIR[-6:])

        maybe_mkdir_p(path_imagesTest)
        maybe_mkdir_p(path_outputTest)

        if MODE == 'TEST' or MODE == 'EVAL' or MODE == 'TRAIN_TEST' or MODE == 'TRAIN_EVAL':
            model = os.path.join(MODEL_DIR, 'model_best.model')
            pkl   = os.path.join(MODEL_DIR, 'model_best.model.pkl')
            plans = os.path.join(MODEL_DIR[:-7], 'plans.pkl')

            if LOWRES:
                path_models = os.path.join(nnunet_dir, f'nnUNet_base/nnUNet_training_output_dir/nnUNet/3d_lowres/Task100_grid/nnUNetTrainerV2__nnUNetPlansv2.1/{fold}')
            else:
                path_models = os.path.join(nnunet_dir, f'nnUNet_base/nnUNet_training_output_dir/nnUNet/3d_fullres/Task100_grid/nnUNetTrainerV2__nnUNetPlansv2.1/{fold}')

            maybe_mkdir_p(path_models)

            shutil.copyfile(model, os.path.join(path_models, 'model_final_checkpoint.model'))
            shutil.copyfile(pkl,   os.path.join(path_models, 'model_final_checkpoint.model.pkl'))
            shutil.copyfile(plans, os.path.join(path_models[:-len(fold)], 'plans.pkl'))

            test_dir = os.path.join(MODEL_DIR, 'test')
            maybe_mkdir_p(test_dir)

        elif MODE == 'ENSMEBLE_TEST' or MODE == 'ENSMEBLE_EVAL' or MODE == 'ENSMEBLE_TRAIN_TEST' or MODE == 'ENSMEBLE_TRAIN_EVAL':
            test_dir = os.path.join(MODEL_DIR, 'test_ensemble')
            maybe_mkdir_p(test_dir)
        
        if MODE == 'TRAIN_TEST' or MODE == 'TRAIN_EVAL':
            test_dir_labels = os.path.join(test_dir, 'data_training', 'labels')
            test_dir_preds  = os.path.join(test_dir, 'data_training', 'preds' )
        elif MODE == 'ENSEMBLE_TRAIN_TEST' or MODE == 'ENSEMBLE_TRAIN_EVAL':    
            test_dir_labels = os.path.join(test_dir, 'data_training', 'labels')
            test_dir_preds  = os.path.join(test_dir, 'data_training', 'preds_ensemble')
        elif MODE == 'TEST' or MODE == 'EVAL':
            test_dir_labels = os.path.join(test_dir, 'data', 'labels')
            test_dir_preds  = os.path.join(test_dir, 'data', 'preds' )
        elif MODE == 'ENSEMBLE_TEST' or MODE == 'ENSEMBLE_EVAL':    
            test_dir_labels = os.path.join(test_dir, 'data', 'labels')
            test_dir_preds  = os.path.join(test_dir, 'data', 'preds_ensemble')

        maybe_mkdir_p(test_dir_labels)
        maybe_mkdir_p(test_dir_preds)

        # Copy selected model to nnunet output dir
        print('Copying selected model to nnUNet output dir for inference...')

        print('done')
        print('    ')

        if LOW_RAM: 
            print('Using LOW_RAM option (See main.py -h for more information)')
            print('                                                          ')
            path_lowram = os.path.join(path_imagesTest, 'low_ram_' + trainer + '_' + MODEL_DIR[-6:])
            maybe_mkdir_p(path_lowram)

        # Extract test images and labels from dataset.json
        print('Extracting test images and labels...')
        list_imagesTest = []; list_labelsTest = []
        print(os.path.join(MODEL_DIR[:-6], 'dataset.json'))
        with open(os.path.join(MODEL_DIR[:-6], 'dataset.json')) as json_file:
            data = json.load(json_file)
            if MODE == 'TRAIN_TEST' or MODE == 'ENSEMBLE_TRAIN_TEST':
                if DATASET_CONFIG == 0:
                    for image in data[f'fold {FOLD}']['training']:
                        list_imagesTest.append(image['image'][-15:])
                        list_labelsTest.append(image['image'][-15:])
                elif DATASET_CONFIG == 1:
                    for image in data[f'fold {FOLD}']['training']:
                        list_imagesTest.append(image['image'][-15:])
                        list_labelsTest.append(image['image'][-15:])
            elif MODE == 'TEST' or MODE == 'ENSEMBLE_TEST':
                if DATASET_CONFIG == 0:
                    for image in data[f'fold {FOLD}']['test']:
                        list_imagesTest.append(image['image'][-15:])
                        list_labelsTest.append(image['image'][-15:])
                elif DATASET_CONFIG == 1:
                    for image in data[f'fold {FOLD}']['test']:
                        list_imagesTest.append(image['image'][-15:])
                        list_labelsTest.append(image['image'][-15:])

        list_imagesTest = sorted(list_imagesTest)
        list_labelsTest = sorted(list_labelsTest)

        if CONTINUE:
            list_imagesTest_aux = []
            for pred in list_imagesTest:
                if pred.endswith('.nii.gz') and pred not in os.listdir(path_outputTest):
                    list_imagesTest_aux.append(pred)

            list_imagesTest = list_imagesTest_aux

            if len(list_imagesTest) == 0:
                testing_done = True
                print('Testing is done, going to evaluation')
                print('                                    ')
            else:
                print('Continuing from image', list_imagesTest[0])
                print(len(list_imagesTest), 'images to go'       )
                print('                                         ')

        # Copy testing images and labels to testing dirs
        for image in list_imagesTest:
            if image.endswith('.nii.gz') and f'{image[-15:-7]}_0000.nii.gz' not in os.listdir(path_imagesTest):
                shutil.copyfile(os.path.join(path_images_base, image[-15:]), os.path.join(path_imagesTest, f'{image[-15:-7]}_0000.nii.gz'))

        print('done')
        print('    ')

        ################################# Inference ########################################## 

        start = time()

        if MODE == 'TEST' or MODE == 'TRAIN_TEST':
            if LOW_RAM and not testing_done:
                if path_imagesTest[:8] == '/content': # We are working on drive, we need to get rid of spaces in the path
                    path_outputTest = path_outputTest[:17] + '\ ' + path_outputTest[18:]
                for image in list_imagesTest:
                    image = image[-15:-7] + '_0000.nii.gz'
                    # Remove preexisting files from auxiliar dir
                    for files in glob(os.path.join(path_lowram, '*')):
                        os.remove(files)
                    print('Performing inference over', image)
                    
                    # Copy a single image to auxiliar dir
                    shutil.copyfile(os.path.join(path_imagesTest, image), os.path.join(path_lowram, image))

                    if path_imagesTest[:8] == '/content':
                        path_lowram = path_lowram[:17] + '\ ' + path_lowram[18:]

                    if LOWRES:
                        os.system('nnUNet_predict -i ' + path_lowram + ' -o ' + path_outputTest + f' -t Task100_grid -m 3d_lowres -f ' + TEST_FOLD)
                    else:
                        os.system('nnUNet_predict -i ' + path_lowram + ' -o ' + path_outputTest + f' -t Task100_grid -m 3d_fullres -f ' + TEST_FOLD)

                    if path_imagesTest[:8] == '/content':
                        path_lowram = os.path.join(path_imagesTest, 'low_ram_' + trainer + '_' + MODEL_DIR[-6:])

                        print(f'Inference over {image} finished')
                        print('                                ')
            elif not LOW_RAM and not testing_done:
                if path_imagesTest[:8] == '/content': # We are working on drive, we need to get rid of spaces in the path
                    path_imagesTest = path_imagesTest[:17] + '\ ' + path_imagesTest[18:]
                    path_outputTest = path_outputTest[:17] + '\ ' + path_outputTest[18:]

                print('Performing inference with the testing set:')
                print('                                          ')
                if LOWRES:
                    os.system('nnUNet_predict -i ' + path_imagesTest + ' -o ' + path_outputTest + f' -t Task100_grid -m 3d_lowres -f ' + TEST_FOLD)
                else:
                    os.system('nnUNet_predict -i ' + path_imagesTest + ' -o ' + path_outputTest + f' -t Task100_grid -m 3d_fullres -f ' + TEST_FOLD)

                print('                  ')
                print('Inference finished')

        elif MODE == 'ENSEMBLE_TEST' or MODE  == 'ENSEMBLE_TRAIN_TEST':
            if not testing_done:
                if path_imagesTest[:8] == '/content': # We are working on drive, we need to get rid of spaces in the path
                    path_outputTest = path_outputTest[:17] + '\ ' + path_outputTest[18:]

            for image in list_imagesTest:
                # Define paths for ensemble
                INPUT_DIR = os.path.join(path_imagesTest, image[-15:-7])
                INPUT = image[-15:]
                maybe_mkdir_p(INPUT_DIR)

                # Insert input file in input dir
                shutil.copyfile(os.path.join(path_imagesTest, f'{image[-15:-7]}_0000.nii.gz'), os.path.join(INPUT_DIR, INPUT))
                
                # Perform ensembled inference (only with LOW_RAM=True)
                ensemble_single(nnunet_dir=nnunet_dir, INPUT_DIR=INPUT_DIR, INPUT=INPUT, MODEL_DIR=MODEL_DIR, MODEL=MODEL, LOWRES=LOWRES)

                # Moves output nifti to output dir in the proper format
                os.rename(os.path.join(INPUT_DIR, 'output', INPUT), os.path.join(path_outputTest, INPUT))

            print(f'Inference with ensembling took {time() - start} s ({len(list_imagesTest)} images)')
            print('                                                                                  ')

        # Move all inferred images and labels to test dir
        print('Moving all labels and predictions to:')
        print(test_dir                               )
        path_outputTest = os.path.join(nnunet_dir, 'inference_test/output_' + trainer + '_' + MODEL_DIR[-6:])
        for label in list_labelsTest:
            shutil.copyfile(os.path.join(path_labels_base, label[-15:]), os.path.join(test_dir_labels, label[-15:]))
        for pred  in os.listdir(path_outputTest):
            if pred.endswith('nii.gz'):
                os.rename(os.path.join(path_outputTest, pred),  os.path.join(test_dir_preds,  pred ))

        print('done')
        print('    ')

    if MODE in ['TEST', 'EVAL', 'TRAIN_TEST', 'TRAIN_EVAL', 'ENSEMBLE_TEST', 'ENSMEBLE_EVAL', 'ENSEMBLE_TRAIN_TEST', 'ENSEMBLE_TRAIN_EVAL']:
        ################################ Evaluation ##########################################

        # Perform evaluation over inferred samples
        eval_metrics(MODEL_DIR, MODE, CONTINUE)

    print('End of testing')



##############################################################################################
#--------------------------------------------------------------------------------------------#    
##############################################################################################