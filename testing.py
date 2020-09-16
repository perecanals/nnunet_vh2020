import numpy as np

import os
from glob import glob
import shutil
import json
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

from evaluation_metrics import eval_metrics

from time import time

def testing(nnunet_dir, MODEL_DIR=None, MODE=None, LOW_RAM=None, MODEL=None, LOWRES=False, trainer='nnUNetTrainerV2', CONTINUE=False, TEST_FOLD = 0):
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

    if TEST_FOLD not in [0, 1, 2, 3, 4]:
        TEST_FOLD = 'all'
        fold = 'all'
    else:
        TEST_FOLD = str(TEST_FOLD)
        fold = 'fold_' + TEST_FOLD

    testing_done = False

    if MODE == 'TESTING' or MODE == 'INFERENCE' or MODE == 'TRAIN_TEST' or MODE == 'TEST_ALL':

        ################################## Testing ###########################################

        # Paths
        path_images_base = os.path.join(nnunet_dir, 'database_vh/database_images')
        path_labels_base = os.path.join(nnunet_dir, 'database_vh/database_labels')

        maybe_mkdir_p(os.path.join(nnunet_dir, 'inference_test'))

        path_imagesTest = os.path.join(nnunet_dir, 'inference_test/input'                                   )
        path_labelsTest = os.path.join(nnunet_dir, 'inference_test/labels'                                  )
        path_outputTest = os.path.join(nnunet_dir, 'inference_test/output_' + trainer + '_' + MODEL_DIR[-6:])

        maybe_mkdir_p(path_imagesTest)
        maybe_mkdir_p(path_labelsTest)
        maybe_mkdir_p(path_outputTest)

        model = os.path.join(MODEL_DIR, 'model_best.model')
        pkl   = os.path.join(MODEL_DIR, 'model_best.model.pkl')
        plans = os.path.join(MODEL_DIR[:-7], 'plans.pkl')

        if LOWRES:
            path_models = os.path.join(nnunet_dir, f'nnUNet_base/nnUNet_training_output_dir/nnUNet/3d_lowres/Task100_grid/nnUNetTrainerV2__nnUNetPlansv2.1/{fold}')
        else:
            path_models = os.path.join(nnunet_dir, f'nnUNet_base/nnUNet_training_output_dir/nnUNet/3d_fullres/Task100_grid/nnUNetTrainerV2__nnUNetPlansv2.1/{fold}')

        maybe_mkdir_p(path_models)

        shutil.copyfile(plans, os.path.join(path_models[:-len(fold)], 'plans.pkl'))

        test_dir = os.path.join(MODEL_DIR, 'test')
        maybe_mkdir_p(test_dir)

        if MODE == 'TRAIN_TEST':
            test_dir_labels = os.path.join(test_dir, 'data_training', 'labels')
            test_dir_preds  = os.path.join(test_dir, 'data_training', 'preds' )
        else:
            test_dir_labels = os.path.join(test_dir, 'data', 'labels')
            test_dir_preds  = os.path.join(test_dir, 'data', 'preds' )
        maybe_mkdir_p(test_dir_labels)
        maybe_mkdir_p(test_dir_preds)

        # Copy selected model to nnunet output dir
        print('Copying selected model to nnUNet output dir for inference...')

        shutil.copyfile(model, os.path.join(path_models, 'model_final_checkpoint.model'))
        shutil.copyfile(pkl,   os.path.join(path_models, 'model_final_checkpoint.model.pkl'))

        print('done')
        print('    ')

        if LOW_RAM is not None: 
            print('Using LOW_RAM option (See main.py -h for more information)')
            print('                                                          ')
            path_lowram = os.path.join(path_imagesTest, 'low_ram_' + trainer + '_' + MODEL_DIR[-6:])
            maybe_mkdir_p(path_lowram)

        # Extract test images and labels from dataset.json
        print('Extracting test images and labels...')
        list_imagesTest = []; list_labelsTest = []
        if MODE != 'TEST_ALL':
            print(os.path.join(MODEL_DIR[:-6], 'dataset.json'))
            with open(os.path.join(MODEL_DIR[:-6], 'dataset.json')) as json_file:
                data = json.load(json_file)
                if MODE == 'TRAIN_TEST':
                    for image in data['training']:
                        list_imagesTest.append(image['image'][-15:])
                        list_labelsTest.append(image['image'][-15:])
                else:
                    for image in data['test']:
                        list_imagesTest.append(image['image'][-15:])
                        list_labelsTest.append(image['image'][-15:])
        else:
            train_list = []
            with open(os.path.join(MODEL_DIR[:-6], 'dataset.json')) as json_file:
                data = json.load(json_file)
                for image in data['training']:
                    train_list.append(image['image'][-15:])
            for image in os.listdir(path_images_base):
                if image not in train_list and image.endswith('.nii.gz'):
                    list_imagesTest.append(image)
                    list_labelsTest.append(image)

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

        # Remove preexisting nifti files in testing dirs
        for files in glob(os.path.join(path_imagesTest, '*.gz')):
            if (files[-20:-12] + '.nii.gz') not in list_imagesTest:
                os.remove(files)
        for files in glob(os.path.join(path_labelsTest, '*.gz')):
            if files not in list_labelsTest:
                os.remove(files)

        # Copy testing images and labels to testing dirs
        for image in list_imagesTest:
            if image.endswith('.nii.gz') and f'{image[-15:-7]}_0000.nii.gz' not in os.listdir(path_imagesTest):
                shutil.copyfile(os.path.join(path_images_base, image[-15:]), os.path.join(path_imagesTest, f'{image[-15:-7]}_0000.nii.gz'))
        for label in list_labelsTest:
            if label.endswith('.nii.gz') and label not in os.listdir(path_labelsTest):
                shutil.copyfile(os.path.join(path_labels_base, label[-15:]), os.path.join(path_labelsTest, label[-15:]))

        print('done')
        print('    ')

        ################################# Inference ########################################## 

        start = time()

        if LOW_RAM is not None and not testing_done:
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
        elif not testing_done:
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

        print(f'Inference took {time() - start} s ({len(list_imagesTest)} images)')
        print('                                                                  ')

        # Move all inferred images and labels to test dir
        print('Moving all labels and predictions to:')
        print(test_dir                               )
        path_labelsTest = os.path.join(nnunet_dir, 'inference_test/labels'                                  )
        path_outputTest = os.path.join(nnunet_dir, 'inference_test/output_' + trainer + '_' + MODEL_DIR[-6:])
        for label in os.listdir(path_labelsTest):  
            if label.endswith('nii.gz'):
                shutil.copyfile(os.path.join(path_labelsTest, label), os.path.join(test_dir_labels, label))
        for pred  in os.listdir(path_outputTest):
            if pred.endswith('nii.gz'):
                os.rename(os.path.join(path_outputTest, pred),  os.path.join(test_dir_preds,  pred ))

        print('done')
        print('    ')

    if MODE == 'TESTING' or MODE == 'EVALUATION' or MODE == 'TRAIN_TEST' or MODE == 'TRAIN_EVAL' or MODE == 'TEST_ALL' or MODE == 'EVAL_ALL':

        ################################ Evaluation ##########################################

        # Perform evaluation over inferred samples
        eval_metrics(MODEL_DIR, MODE, CONTINUE)

    print('End of testing')



##############################################################################################
#--------------------------------------------------------------------------------------------#    
##############################################################################################