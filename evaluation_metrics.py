import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms

import os
import nibabel as nib
from nibabel.testing import data_path
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

from time import time 

def eval_metrics(nnunet_dir, MODEL_DIR):
    ''' Evaluation metrics computation. This script is used either in 
    TESTING mode or EVALUATION mode. In EVALUATION mode, it assumes that 
    inference has been run over all images belonging to the testing set of
    a model, and  stored in ./nnunet/models/[name_of_model]/test/data.

    Voxel error (ve), intersection over union (iou) and Dice coefficient (dice)
    are all computed in this evaluation.

    Files containing results for all metrics and all prediction, as well as mean 
    and standard deviation are saved in .../test/progress.

    Git repository: https://github.com/perecanals/nnunet_vh2020.git
        
    '''

    start = time()

    # Paths
    test_dir = join(MODEL_DIR, 'test')

    progress_dir = join(test_dir, 'progress')
    maybe_mkdir_p(progress_dir)

    test_dir_labels = join(test_dir, 'data', 'labels')
    test_dir_preds  = join(test_dir, 'data', 'preds' )

    # Retrieve labels and predictions from data dir
    print('Reading data...')
    list_label = []; list_pred = []
    for file in os.listdir(test_dir_labels):
        filename = os.fsdecode(file)
        if filename.endswith(".gz"):
            list_label.append(filename)
            continue
        else:
            continue
    for file in os.listdir(test_dir_preds):
        filename = os.fsdecode(file)
        if filename.endswith(".gz"):
            list_pred.append(filename)
            continue
        else:
            continue
    list_label.sort()
    list_pred.sort()

    print('done')
    print('    ')

    # Initialize evaluation metrics vectors
    ve   = np.empty([len(list_label)])
    iou  = np.empty([len(list_label)])
    dice = np.empty([len(list_label)])

    print('Number of images for testing:', len(list_label))
    print('                                              ')

    for idx in range(len(list_label)):
        # Read niftis
        print(f'Evaluation for {list_pred[idx]} ({list_label[idx]})')
        label = np.array(nib.load(join(test_dir_labels, list_label[idx])).get_data())
        pred  = np.array(nib.load(join(test_dir_preds,  list_pred[idx] )).get_data())

        # Compute evaluation metrics
        ve_idx   = Voxel_error(label, pred)
        iou_idx  = IoU(label, pred)
        dice_idx = Dice(label, pred)

        ve[idx]   = ve_idx
        iou[idx]  = iou_idx
        dice[idx] = dice_idx

        print('Voxel error:', '{:.6f}'.format(ve[idx])  )
        print('IoU:',         '{:.6f}'.format(iou[idx]) )
        print('Dice:',        '{:.6f}'.format(dice[idx]))
        print('                                        ')

    ve_mean   = np.mean(ve)
    ve_std    = np.std(ve)
    iou_mean  = np.mean(iou)
    iou_std   = np.std(iou)
    dice_mean = np.mean(dice)
    dice_std  = np.mean(dice)

    np.savetxt(os.path.join(progress_dir, 've.out'),   ve  )
    np.savetxt(os.path.join(progress_dir, 'iou.out'),  iou )
    np.savetxt(os.path.join(progress_dir, 'dice.out'), dice)

    np.savetxt(os.path.join(progress_dir, 've_mean_std.out'),   [ve_mean,   ve_std]  )
    np.savetxt(os.path.join(progress_dir, 'iou_mean_std.out'),  [iou_mean,  iou_std] )
    np.savetxt(os.path.join(progress_dir, 'dice_mean_std.out'), [dice_mean, dice_std])

    print('Files saved in', progress_dir)
    print('                            ')

    print('Results:                                     ')
    print('Mean voxel error:', '{:.6f}'.format(ve_mean)  )
    print('Mean IoU:',         '{:.6f}'.format(iou_mean) )
    print('Mean Dice:',        '{:.6f}'.format(dice_mean))

    print(f'Evaluation took {time() - start} s ({len(list_label)} images)')
    print('                                                                   ')



##############################################################################################
#--------------------------------------------------------------------------------------------#    
##############################################################################################



def Voxel_error(label, pred):
    ''' Computes the voxel error of the network's prediction compared to the label corresponding to the same image.
        Inputs:
            - pred: prediction. Numpy array.
            - label: label. Numpy array. 

        Output:
            - ve: float.

    '''

    ve = np.sum(abs(pred - label)) / pred.size

    return ve


def IoU(label, pred):
    ''' Intersection over Union (IoU) between labels and predictions.
        Inputs:
            - pred: prediction. Numpy array.
            - label: label. Numpy array. 

        Output:
            - iou: float.

    '''

    intersection = np.logical_and(pred, label)
    union        = np.logical_or(pred,  label)

    iou = np.sum(intersection) / np.sum(union)

    return iou


def Dice(label, pred):
    ''' Dice coefficient (or f1-score, f-measure) between labels and predictions.
        Inputs:
            - pred: prediction. Numpy array.
            - label: label. Numpy array. 

        Output:
            - dice: float.

    '''

    dice = np.sum(pred[label==1]==1)*2.0 / (np.sum(pred==1) + np.sum(label==1))

    return dice



##############################################################################################
#--------------------------------------------------------------------------------------------#    
##############################################################################################