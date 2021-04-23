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

    ### In this case just set always to 1
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
        ### Set paths to inputs, outputs, model (lowram?)

        path_imagesTest = os.path.join(nnunet_dir, 'inference_test/input'                                   )
        path_outputTest = os.path.join(nnunet_dir, 'inference_test/output_' + trainer + '_' + MODEL_DIR[-6:])

        maybe_mkdir_p(path_imagesTest)
        maybe_mkdir_p(path_outputTest)

        if LOWRES:
            path_models = os.path.join(nnunet_dir, f'nnUNet_base/nnUNet_training_output_dir/nnUNet/3d_lowres/Task100_grid/nnUNetTrainerV2__nnUNetPlansv2.1/{fold}')
        else:
            path_models = os.path.join(nnunet_dir, f'nnUNet_base/nnUNet_training_output_dir/nnUNet/3d_fullres/Task100_grid/nnUNetTrainerV2__nnUNetPlansv2.1/{fold}')

        maybe_mkdir_p(path_models)

        if LOW_RAM is not None: 
            print('Using LOW_RAM option (See main.py -h for more information)')
            print('                                                          ')
            path_lowram = os.path.join(path_imagesTest, 'low_ram_' + trainer + '_' + MODEL_DIR[-6:])
            maybe_mkdir_p(path_lowram)

        ### Read images in database

        list_imagesTest = sorted(os.listdir(path_imagesTest))

        ### I recommend to keep track of infered images vs input images in case process gets killed

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

        ################################# Inference ########################################## 

        start = time()

        ### The format on our images is a string of 8 numbers + ".nii.gz" (eg "12345678.nii.gz"). 
        ### Adapt this to your needs or find an alternative solution

        if LOW_RAM is not None and not testing_done: ### If you want to work with one image at the time
            for image in list_imagesTest:

                ### Due to the possibility of several existing modalitites, you have to add the "_0000" !!!
                image = image[-15:-7] + '_0000.nii.gz'

                # Remove preexisting files from auxiliar dir
                for files in glob(os.path.join(path_lowram, '*')):
                    os.remove(files)
                print('Performing inference over', image)
                
                # Copy a single image to auxiliar dir
                shutil.copyfile(os.path.join(path_imagesTest, image), os.path.join(path_lowram, image))

                if LOWRES:
                    os.system('nnUNet_predict -i ' + path_lowram + ' -o ' + path_outputTest + f' -t Task100_grid -m 3d_lowres -f ' + TEST_FOLD)
                else:
                    os.system('nnUNet_predict -i ' + path_lowram + ' -o ' + path_outputTest + f' -t Task100_grid -m 3d_fullres -f ' + TEST_FOLD)

                print(f'Inference over {image} finished')
                print('                                ')

        elif not testing_done and not LOW_RAM: ### If you wanna throw the whole dataset to it
            if LOWRES:
                os.system('nnUNet_predict -i ' + path_imagesTest + ' -o ' + path_outputTest + f' -t Task100_grid -m 3d_lowres -f ' + TEST_FOLD)
            else:
                os.system('nnUNet_predict -i ' + path_imagesTest + ' -o ' + path_outputTest + f' -t Task100_grid -m 3d_fullres -f ' + TEST_FOLD)

            print('                  ')
            print('Inference finished')

        print(f'Inference took {time() - start} s ({len(list_imagesTest)} images)')
        print('                                                                  ')


        print('done')
        print('    ')

    print('End of testing')



##############################################################################################
#--------------------------------------------------------------------------------------------#    
##############################################################################################