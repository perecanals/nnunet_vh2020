import numpy as np

import os
from glob import glob
import shutil
import json
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

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
        - Voxel error
        
    See evaluation_metrics.py for more information.

    Git repository: https://github.com/perecanals/nnunet_vh2020.git
    Original nnunet (Isensee et al. 2020[1]): https://github.com/MIC-DKFZ/nnUNet.git

    [1] Fabian Isensee, Paul F. Jäger, Simon A. A. Kohl, Jens Petersen, Klaus H. Maier-Hein "Automated Design of Deep Learning 
    Methods for Biomedical Image Segmentation" arXiv preprint arXiv:1904.08128 (2020).

    '''

    if MODEL_DIR is None: ValueError('Please input path to model. See main.py -h for more information.')

    if MODE == 'TESTING' or MODE == 'INFERENCE':

        ################################## Testing ###########################################

        # Paths
        path_images_base = os.path.join(nnunet_dir, 'database_vh/database_images')
        path_labels_base = os.path.join(nnunet_dir, 'database_vh/database_labels')

        maybe_mkdir_p(os.path.join(nnunet_dir, 'inference_test'))

        path_imagesTest = os.path.join(nnunet_dir,  'inference_test/input' )
        path_labelsTest = os.path.join(nnunet_dir,  'inference_test/labels')
        path_outputTest = os.path.join(nnunet_dir,  'inference_test/output')

        maybe_mkdir_p(path_imagesTest)
        maybe_mkdir_p(path_labelsTest)
        maybe_mkdir_p(path_outputTest)

        model = os.path.join(MODEL_DIR, 'model_best.model')
        pkl   = os.path.join(MODEL_DIR, 'model_best.model.pkl')

        path_models = os.path.join(nnunet_dir, 'nnUNet_base/nnUNet_training_output_dir/nnUNet/3d_fullres/Task100_grid/nnUNetTrainerV2__nnUNetPlansv2.1/all/')
        maybe_mkdir_p(path_models)

        test_dir = os.path.join(MODEL_DIR, 'test')
        maybe_mkdir_p(test_dir)

        test_dir_labels = os.path.join(test_dir, 'data', 'labels')
        test_dir_preds  = os.path.join(test_dir, 'data', 'preds' )
        maybe_mkdir_p(os.path.join(test_dir, 'data', 'labels'))
        maybe_mkdir_p(os.path.join(test_dir, 'data', 'preds' ))

        # Copy selected model to nnunet output dir
        print('Copying selected model to nnUNet output dir for inference...')

        shutil.copyfile(model, os.path.join(path_models, 'model_final_checkpoint.model'))
        shutil.copyfile(pkl,   os.path.join(path_models, 'model_final_checkpoint.model.pkl'))

        print('done')
        print('    ')

        if LOW_RAM is not None: 
            print('Using LOW_RAM option (See main.py -h for more information)')
            print('                                                          ')
            path_lowram = os.path.join(path_imagesTest, 'low_ram')
            maybe_mkdir_p(path_lowram)

        # Extract test images and labels from dataset.json
        print('Extracting test images and labels...')
        list_imagesTest = []; list_labelsTest = []
        print(os.path.join(MODEL_DIR, 'dataset.json'))
        with open(os.path.join(MODEL_DIR, 'dataset.json')) as json_file:
            data = json.load(json_file)
            for image in data['test']:
                list_imagesTest.append(image['image'][11:])
                list_labelsTest.append(image['image'][11:])

        # Remove preexisting nifti files in testing dirs
        for files in glob(os.path.join(path_imagesTest, '*.gz')):
            os.remove(files)
        for files in glob(os.path.join(path_labelsTest, '*.gz')):
            os.remove(files)

        # Copy testing images and labels to testing dirs
        for image in list_imagesTest:
            shutil.copyfile(os.path.join(path_images_base, image), os.path.join(path_imagesTest, f'{image[:8]}_0000.nii.gz'))
        for label in list_labelsTest:
            shutil.copyfile(os.path.join(path_labels_base, label), os.path.join(path_labelsTest, label))

        print('done')
        print('    ')

        ################################# Inference ########################################## 

        start = time()

        if LOW_RAM is not None:
            for image in list_imagesTest:
                # Remove preexisting files from auxiliar dir
                for files in glob(os.path.join(path_lowram, '*.gz')):
                    os.remove(files)
                # Copy a single image to auxiliar dir
                shutil.copyfile(os.path.join(path_imagesTest, f'{image[:8]}_0000.nii.gz'), os.path.join(path_lowram, f'{image[:8]}_0000.nii.gz'))

                print('Performing inference over', image)
                print('                                ')
                os.system('nnUNet_predict -i ' + path_lowram + ' -o ' + path_outputTest + ' -t Task100_grid -m 3d_fullres -f all --save_npz')

                print('                                ')
                print(f'Inference over {image} finished')
        else:
            print('Performing inference with the whole testing set:')
            print('                                                ')
            os.system('nnUNet_predict -i ' + path_imagesTest + ' -o ' + path_outputTest + ' -t Task100_grid -m 3d_fullres -f all --save_npz')

            print('                  ')
            print('Inference finished')

        print(f'Inference took {time() - start} s ({len(list_imagesTest)} images)')
        print('                                                                  ')

        # Move all inferred images and labels to test dir
        print('Moving all labels and predictions to:')
        print(test_dir                               )
        for label in glob(os.path.join(path_labelsTest, '*.gz')):  
            os.rename(os.path.join(path_labelsTest, label), os.path.join(test_dir_labels, label))
        for pred  in glob(os.path.join(path_outputTest, '*.gz')):
            os.rename(os.path.join(path_outputTest, pred),  os.path.join(test_dir_preds,  pred ))

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