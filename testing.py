import numpy as np

import os
import glob
import shutil
import json
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

from evaluation_metrics import eval_metrics

from time import time

def testing(nnunet_dir, MODEL_DIR=None, MODE=None, LOW_RAM=None):
    ''' ################################ Testing #############################################

    Testing of the nnunet trained models. All images and labels in the 
    imagesTs section in the dataset.json in the model's directory
    should be in the database directories. The dataset.json file should
    be generated in advance, when performing training, and stored in 
    MODEL_DIR.

    Evaluation metrics computed: 
        - IoU
        - Dice coefficient
        - F1-score. 
        
    See evaluation_metrics.py for more information.

    Git repository: https://github.com/perecanals/nnunet_vh2020.git
    Original nnunet (Isensee et al. 2019): https://github.com/MIC-DKFZ/nnUNet.git

    '''

    if MODEL_DIR is None: ValueError('Please input path to model. See main.py -h for more information.')

    if MODE == 'TESTING' or MODE == 'INFERENCE':

        ################################## Testing ###########################################

        # Paths
        path_images_base = join(nnunet_dir, 'database_vh/database_images')
        path_labels_base = join(nnunet_dir, 'database_vh/database_labels')

        path_imagesTest = join(nnunet_dir,  'inference_test/input' )
        path_labelsTest = join(nnunet_dir,  'inference_test/labels')
        path_outputTest = join(nnunet_dir,  'inference_test/output')

        model = join(MODEL_DIR, 'model_best.model')
        pkl   = join(MODEL_DIR, 'model_best.pkl')

        path_models = join(nnunet_dir, 'nnUNet_base/nnUNet_training_output_dir/3d_fullres/Task00_grid/nnUNetTrainer__nnUNetPlans/all/')

        test_dir = join(MODEL_DIR, 'test')
        maybe_mkdir_p(test_dir)

        test_dir_labels = join(test_dir, 'data', 'labels')
        test_dir_preds  = join(test_dir, 'data', 'preds' )
        maybe_mkdir_p(join(test_dir, 'data', 'labels'))
        maybe_mkdir_p(join(test_dir, 'data', 'preds' ))

        # Copy selected model to nnunet output dir
        print('Copying selected model to nnUNet output dir for inference...')
        shutil.copyfile(model, path_models)
        shutil.copyfile(pkl,   path_models)

        print('done')
        print('    ')

        if LOW_RAM is not None: 
            print('Using LOW_RAM option (See main.py -h for more information)')
            print('                                                          ')
            path_lowram = join(path_imagesTest, 'low_ram')
            maybe_mkdir_p(path_lowram)

        # Extract test images and labels from dataset.json
        print('Extracting test images and labels...')
        list_imagesTest = []; list_labelsTest = []
        with open('dataset.json') as json_file:
            data = json.load(json_file)
            for image in data['test']:
                list_imagesTest.append(image[11:])
                list_labelsTest.append(image[:8] + 'L' + image[8:])

        # Remove preexisting nifti files in testing dirs
        for files in glob.glob(join(path_imagesTest, '*.gz')):
            os.remove(files)
        for files in glob.glob(join(path_labelsTest, '*.gz')):
            os.remove(files)

        # Copy testing images and labels to testing dirs
        for image in list_imagesTest:
            shutil.copyfile(join(path_images_base, image), join(path_imagesTest, image))
        for label in list_labelsTest:
            shutil.copyfile(join(path_labels_base, label), join(path_labelsTest, label))

        print('done')
        print('    ')

        ################################# Inference ########################################## 

        start = time()

        if LOW_RAM is not None:
            print('Performing inference with the whole testing set:')
            print('                                                ')
            os.system("OMP_NUM_THREADS=1 python3 inference/predict_simple.py -i " + path_imagesTest + " -o " + path_outputTest + " -t Task00_grid -tr nnUNetTrainer -m 3d_fullres -f all")

            print('                  ')
            print('Inference finished')
        else:
            for image in list_imagesTest:
                # Remove preexisting files from auxiliar dir
                for files in glob.glob(join(path_lowram, '*.gz')):
                    os.remove(files)
                # Copy a single image to auxiliar dir
                shutil.copyfile(join(path_imagesTest, image), join(path_lowram, image))

                print('Performing inference over', image)
                print('                                ')
                os.system("OMP_NUM_THREADS=1 python3 inference/predict_simple.py -i " + path_lowram + " -o " + path_outputTest + " -t Task00_grid -tr nnUNetTrainer -m 3d_fullres -f all")

                print('                                ')
                print(f'Inference over {image} finished')

        print(f'Inference took {time() - start} s ({len(list_imagesTest)} images)')
        print('                                                                  ')

        # Move all inferred images and labels to test dir
        print('Moving all labels and predictions to:')
        print(test_dir                               )
        for label in glob.glob(join(path_labelsTest, '*.gz')):  
            os.rename(join(path_labelsTest, label), join(test_dir_labels, label))
        for pred  in glob.glob(join(path_outputTest, '*.gz')):
            os.rename(join(path_outputTest, pred),  join(test_dir_preds,  pred ))

            print('done')
            print('    ')

    if MODE == 'TESTING' or MODE == 'EVALUATION':

        ################################ Evaluation ##########################################

        # Perform evaluation over inferred samples
        eval_metrics(nnunet_dir, MODEL_DIR)

    print('End of testing')



##############################################################################################
#--------------------------------------------------------------------------------------------#    
##############################################################################################