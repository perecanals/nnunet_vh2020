import os
from glob import glob
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
import numpy as np

import nibabel as nib
from nibabel.testing import data_path
import torch
import torchvision
import torchvision.transforms as transforms

from medpy.metric.binary import jc, volume_correlation, precision, recall 

from time import time 

def eval_metrics(nnunet_dir, MODEL_DIR, MODE=None):
    ''' Evaluation metrics computation. This script is used either in 
    TESTING, EVALUATION, TRAIN_TEST or TRAIN_EVAL modes. In EVALUATION mode, it assumes that 
    inference has been run over all images belonging to the testing set of
    a model, and stored in ./nnunet/models/[name_of_model]/test/data.

    Dice coefficient (dice), Jaccard index (jacc), Precision (prec) and Recall (reca)
    are all computed in this evaluation.

    Files containing results for all metrics and all prediction, as well as mean 
    and standard deviation are saved in .../test/progress.

    Git repository: https://github.com/perecanals/nnunet_vh2020.git
        
    '''

    if MODE == 'TESTING' or MODE == 'EVALUATION':
        mode = 'test'
    elif MODE == 'TRAIN_TEST' or MODE == 'TRAIN_EVAL':
        mode = 'train'
    elif MODE == 'TEST_ALL' or 'EVAL_ALL':
        mode = 'test_all'

    start = time()

    # Paths
    test_dir = os.path.join(MODEL_DIR, 'test')

    progress_dir = os.path.join(test_dir, 'progress', mode)
    maybe_mkdir_p(progress_dir)

    if MODE == 'TRAIN_TEST' or MODE == 'TRAIN_EVAL':
        test_dir_labels = os.path.join(test_dir, 'data_training', 'labels')
        test_dir_preds  = os.path.join(test_dir, 'data_training', 'preds' )
    else:
        test_dir_labels = os.path.join(test_dir, 'data', 'labels')
        test_dir_preds  = os.path.join(test_dir, 'data', 'preds' )

    # Retrieve labels and predictions from data dir
    print('Reading data...')

    list_label = sorted(glob(os.path.join(test_dir_labels, '*.nii.gz')))
    list_pred  = sorted(glob(os.path.join(test_dir_preds , '*.nii.gz')))

    assert len(list_label) == len(list_pred)

    print('done')
    print('    ')

    # Initialize evaluation metrics vectors
    dice = np.empty([len(list_label)])
    jacc = np.empty([len(list_label)])
    prec = np.empty([len(list_label)])
    reca = np.empty([len(list_label)])

    print('Number of images for testing:', len(list_label))
    print('                                              ')

    for idx in range(len(list_label)):
        # Read niftis
        print(f'Evaluation for {list_pred[idx][-15:]}')
        label = nib.load(list_label[idx]).get_data()
        pred  = nib.load(list_pred[idx] ).get_data()

        # Compute evaluation metrics
        dice_idx = Dice(label, pred)
        jacc_idx = jc(label, pred)
        prec_idx = precision(label, pred)
        reca_idx = recall(label, pred)

        dice[idx] = dice_idx
        jacc[idx] = jacc_idx
        prec[idx] = prec_idx
        reca[idx] = reca_idx

        print('Dice     :', '{:.6f}'.format(dice[idx]))
        print('Jaccard  :', '{:.6f}'.format(jacc[idx]))
        print('Precision:', '{:.6f}'.format(prec[idx]))
        print('Recall   :', '{:.6f}'.format(reca[idx]))
        print('                                      ')

    dice_mean = np.mean(dice)
    dice_std  = np.std(dice)
    jacc_mean = np.mean(jacc)
    jacc_std  = np.std(jacc)
    prec_mean = np.mean(prec)
    prec_std  = np.std(prec)
    reca_mean = np.mean(reca)
    reca_std  = np.std(reca)

    np.savetxt(os.path.os.path.join(progress_dir, 'dice.out')     , dice)
    np.savetxt(os.path.os.path.join(progress_dir, 'jaccard.out')  , jacc)
    np.savetxt(os.path.os.path.join(progress_dir, 'precision.out'), prec)
    np.savetxt(os.path.os.path.join(progress_dir, 'recall.out')   , reca)

    np.savetxt(os.path.os.path.join(progress_dir, 'dice_mean_std.out')     , [dice_mean, dice_std])
    np.savetxt(os.path.os.path.join(progress_dir, 'jaccard_mean_std.out')  , [jacc_mean, jacc_std])
    np.savetxt(os.path.os.path.join(progress_dir, 'precision_mean_std.out'), [prec_mean, prec_std])
    np.savetxt(os.path.os.path.join(progress_dir, 'recall_mean_std.out')   , [reca_mean, reca_std])

    print('Files saved in', progress_dir)
    print('                            ')

    print('Results:                                   ')
    print('Mean Dice     :', '{:.6f}'.format(dice_mean))
    print('Mean Jaccard  :', '{:.6f}'.format(jacc_mean))
    print('Mean Precision:', '{:.6f}'.format(prec_mean))
    print('Mean Recall   :', '{:.6f}'.format(reca_mean))

    print(f'Evaluation took {time() - start} s ({len(list_label)} images)')
    print('                                                              ')


##############################################################################################
#--------------------------------------------------------------------------------------------#
##############################################################################################


def individual_eval(pred, label):
    ''' Input path to nifti

    Dice coefficient (dice), Jaccard index (jacc), Precision (prec) and Recall (reca)
    are all computed in this evaluation.

    Files containing results for all metrics and all prediction, as well as mean 
    and standard deviation are saved in .../test/progress.

    Git repository: https://github.com/perecanals/nnunet_vh2020.git
        
    '''

    pat_id = pred[-15:]

    label = nib.load(label).get_data()
    pred  = nib.load(pred ).get_data()

    # Compute evaluation metrics
    print(f'Evaluation for {pat_id}')
    dice = Dice(label, pred)
    jacc = jc(label, pred)
    prec = precision(label, pred)
    reca = recall(label, pred)

    print('Dice     :', '{:.6f}'.format(dice))
    print('Jaccard  :', '{:.6f}'.format(jacc))
    print('Precision:', '{:.6f}'.format(prec))
    print('Recall   :', '{:.6f}'.format(reca))
    print('                                 ')


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

    smooth = 1e-6

    iou = np.sum(intersection) / (np.sum(union) + smooth)

    return iou


def Dice(label, pred):
    ''' Dice coefficient (or f1-score, f-measure) between labels and predictions.
        Inputs:
            - pred: prediction. Numpy array.
            - label: label. Numpy array. 

        Output:
            - dice: float.

    '''

    intersection = np.sum(np.logical_and(pred, label))
    union        = np.sum(np.logical_or(pred,  label))

    smooth = 1e-6

    dice = 2 * intersection / (union + intersection + smooth)

    return dice


from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure

def surface_distance(pred, label, voxel_size):
    ''' ####################### Voxel-wise Hausdorff distance ##################################

    Computed voxel-wise Hausdorff Distance of border pixels of predicted volume. It does so by:
        1) Passes label and prediction to binary.
        2) Computes volumes' surfaces with scipy.ndimage.morphology.binary_erosion
        3) Computes distance transform of negated label.
        4) Only maintains voxels that belong to prediction's border, keeping original master volume.*
            * Addition to original code (from medpy.metric.binary.__surface_distances)

        Inputs:
            - pred: predicted output, binary mask from the network. Numpy array [H, W, D] with ones and zeros.
            - label: ground truth. Numpy array [H, W, D] with ones and zeros.
            - voxel_size: voxel size of original image.

        Returns:
            - surf_dist: hausdorff distance of prediction's border voxels with original master volume. Numpy array
            [H, W, D] with numerical values.
            - sds: hausdorff distance of prediction's border voxels as a 1D numpy array (only non-zero values).

    '''

    pred = np.atleast_1d(pred.astype(np.bool))
    label = np.atleast_1d(label.astype(np.bool))

    voxel_size = _ni_support._normalize_sequence(voxel_size, pred.ndim)
    voxel_size = np.asarray(voxel_size, dtype=np.float64)

    footprint = generate_binary_structure(pred.ndim, 1)

    pred_border = pred ^ binary_erosion(pred, structure=footprint, iterations=1)
    label_border = label ^ binary_erosion(label, structure=footprint, iterations=1)

    dt = distance_transform_edt(~label_border, sampling=voxel_size)

    sds = dt[pred_border]

    surf_dist = dt * pred_border # Addition to the original code

    return surf_dist, sds


def voxel_distance(pred, label, voxel_size):
    ''' ####################### Voxel-wise Hausdorff distance ##################################

    Computed voxel-wise Hausdorff Distance of border pixels of predicted volume. It does so by:
        1) Passes label and prediction to binary.
        2) Computes distance transform of negated label.
        3) Only maintains voxels that belong to prediction, keeping original master volume.*
            * Addition to original code (from medpy.metric.binary.__surface_distances)

        Inputs:
            - pred: predicted output, binary mask from the network. Numpy array [H, W, D] with ones and zeros.
            - label: ground truth. Numpy array [H, W, D] with ones and zeros.
            - voxel_size: voxel size of original image.

        Returns:
            - vox_dist: hausdorff distance of prediction's voxels with original master volume. Numpy array
            [H, W, D] with numerical values.

    '''

    pred = np.atleast_1d(pred.astype(np.bool))
    label = np.atleast_1d(label.astype(np.bool))

    voxel_size = _ni_support._normalize_sequence(voxel_size, pred.ndim)
    voxel_size = np.asarray(voxel_size, dtype=np.float64)

    dt = distance_transform_edt(~label, sampling=voxel_size)

    vox_dist = dt * pred # Addition to the original code

    return vox_dist


def voxel_distance_with_underseg(pred, label, voxel_size):
    ''' ####################### Voxel-wise Hausdorff distance ##################################

    Computed voxel-wise Hausdorff Distance of tp and fn voxels of predicted volume. It does so by:
        1) Passes label and prediction to binary.
        2) Computes distance transform of negated label and prediciton.
        3) Only maintains voxels that belong to pred (tp and fp) and fn's, keeping original master volume.*
            * Addition to original code (from medpy.metric.binary.__surface_distances)

        Inputs:
            - pred: predicted output, binary mask from the network. Numpy array [H, W, D] with ones and zeros.
            - label: ground truth. Numpy array [H, W, D] with ones and zeros.
            - voxel_size: voxel size of original image.

        Returns:
            - vox_dist: output volume formed by distance (mm) to label in voxels belonging to 
            prediction (tp and fp), and distance to prediction of fn voxels (undersegmentation, 
            negative values).

    '''

    pred = np.atleast_1d(pred.astype(np.bool))
    label = np.atleast_1d(label.astype(np.bool))

    voxel_size = _ni_support._normalize_sequence(voxel_size, pred.ndim)
    voxel_size = np.asarray(voxel_size, dtype=np.float64)

    dt = distance_transform_edt(~label, sampling=voxel_size)
    dt2 = distance_transform_edt(~pred, sampling=voxel_size)

    fn = np.logical_and(label, np.logical_or(label, pred)) # Only fn voxels (check)

    vox_dist = dt * pred # Addition to the original code
    underseg = dt2 * fn # Undersegmentation (check): set to negative distances for better identification

    vox_dist = vox_dist - underseg

    return vox_dist

##############################################################################################
#--------------------------------------------------------------------------------------------#    
##############################################################################################