import os
from glob import glob
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
import numpy as np

import nibabel as nib
from nibabel.testing import data_path
import torch
import torchvision
import torchvision.transforms as transforms

from medpy.metric.binary import jc, precision, recall, sensitivity, specificity, true_negative_rate, true_positive_rate, positive_predictive_value, hd, hd95, assd, asd, ravd, volume_correlation, volume_change_correlation, obj_assd, obj_asd, obj_fpr, obj_tpr

from time import time 

def eval_metrics(MODEL_DIR, MODE=None, CONTINUE=False):
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
    dice = np.zeros([len(list_label)])
    jacc = np.zeros([len(list_label)])
    # prec = np.zeros([len(list_label)])
    reca = np.zeros([len(list_label)])
    ravdi = np.zeros([len(list_label)])
    # haud = np.zeros([len(list_label)])
    # haud95 = np.zeros([len(list_label)])
    assdi = np.zeros([len(list_label)])
    # asdi = np.zeros([len(list_label)])
    volc = np.zeros([len(list_label)])
    volcc = np.zeros([len(list_label)])

    print('Number of images for testing:', len(list_label))
    print('                                              ')
    
    n = 0

    if CONTINUE and MODE in ['EVAL_ALL', 'EVALUATION', 'TRAIN_EVAL']:
        dice_continue = np.asarray(np.loadtxt(os.path.join(progress_dir, 'dice.out'))) # Changed (precision -> dice)
        # hausdorff95_continue = np.asarray(np.loadtxt(os.path.join(progress_dir, 'hausdorff95.out'))) # Changed (hausdorff -> hausdorff95)
        for idx, _ in enumerate(dice_continue):
            if dice_continue[idx] > 0.01:
                n += 1
        dice[:n]   = np.asarray(np.loadtxt(os.path.join(progress_dir, 'dice.out'))       )[:n]
        jacc[:n]   = np.asarray(np.loadtxt(os.path.join(progress_dir, 'jaccard.out'))    )[:n]
        # prec[:n]   = np.asarray(np.loadtxt(os.path.join(progress_dir, 'precision.out'))  )[:n]
        reca[:n]   = np.asarray(np.loadtxt(os.path.join(progress_dir, 'recall.out'))     )[:n]
        ravdi[:n]  = np.asarray(np.loadtxt(os.path.join(progress_dir, 'ravd.out'))       )[:n]
        # haud[:n]   = np.asarray(np.loadtxt(os.path.join(progress_dir, 'hausdorff.out'))  )[:n]
        # haud95[:n] = np.asarray(np.loadtxt(os.path.join(progress_dir, 'hausdorff95.out')))[:n]
        assdi[:n]  = np.asarray(np.loadtxt(os.path.join(progress_dir, 'assd.out'))       )[:n]
        # asdi[:n]   = np.asarray(np.loadtxt(os.path.join(progress_dir, 'asd.out'))        )[:n]
        volc[:n]   = np.asarray(np.loadtxt(os.path.join(progress_dir, 'volc.out'))       )[:n]
        volcc[:n]  = np.asarray(np.loadtxt(os.path.join(progress_dir, 'volcc.out'))      )[:n]
        
        print('Continuing from image', list_pred[n][-15:])
        print(len(list_label) - n, 'images to go        ')
        print('                                         ')

    for idx in range(len(list_label))[n:]:
        # Read niftis
        print(f'Evaluation for {list_pred[idx][-15:]}')
        label = nib.load(list_label[idx]).get_data()
        pred  = nib.load(list_pred[idx] ).get_data()
        voxel_spacing = nib.load(list_label[idx]).header['pixdim'][1:4]

        # Compute evaluation metrics
        dice_idx = Dice(label, pred)
        jacc_idx = jc(pred, label)
        # prec_idx = precision(pred, label)
        reca_idx = recall(pred, label)
        ravd_idx = np.abs(ravd(pred, label))
        # haud_idx = hd(pred, label, voxelspacing=voxel_spacing, connectivity=1)
        # haud95_idx = hd95(pred, label, voxelspacing=voxel_spacing, connectivity=1)
        assdi_idx, asdi_idx = assd_asd(pred, label, voxelspacing=voxel_spacing, connectivity=1)
        volc_idx = volume_correlation(pred, label)[0]
        volcc_idx = volume_change_correlation(pred, label)[0]

        dice[idx] = dice_idx
        jacc[idx] = jacc_idx
        # prec[idx] = prec_idx
        reca[idx] = reca_idx
        ravdi[idx] = ravd_idx
        # haud[idx] = haud_idx
        # haud95[idx] = haud95_idx
        assdi[idx] = assdi_idx
        # asdi[idx] = asdi_idx
        volc[idx] = volc_idx
        volcc[idx] = volcc_idx

        print('Dice                               :', '{:.6f}'.format(dice[idx])  )
        print('Jaccard                            :', '{:.6f}'.format(jacc[idx])  )
        # print('Precision                          :', '{:.6f}'.format(prec[idx])  )
        print('Recall                             :', '{:.6f}'.format(reca[idx])  )
        print('Relative absolute volume difference:', '{:.6f}'.format(ravdi[idx]) )
        # print('Hausdorff distance                 :', '{:.6f}'.format(haud[idx])  )
        # print('Hausdorff distance 95%             :', '{:.6f}'.format(haud95[idx]))
        print('Average symetric surface distance  :', '{:.6f}'.format(assdi[idx]) )
        # print('Average surface distance           :', '{:.6f}'.format(asdi[idx])  )
        print('Volume correlation                 :', '{:.6f}'.format(volc[idx])  )
        print('Volume change correlation          :', '{:.6f}'.format(volcc[idx]) )
        print('                                                                  ')

        np.savetxt(os.path.join(progress_dir, 'dice.out')       , dice  )
        np.savetxt(os.path.join(progress_dir, 'jaccard.out')    , jacc  )
        # np.savetxt(os.path.join(progress_dir, 'precision.out')  , prec  )
        np.savetxt(os.path.join(progress_dir, 'recall.out')     , reca  )
        np.savetxt(os.path.join(progress_dir, 'ravd.out')       , ravdi )
        # np.savetxt(os.path.join(progress_dir, 'hausdorff.out')  , haud  )
        # np.savetxt(os.path.join(progress_dir, 'hausdorff95.out'), haud95)
        np.savetxt(os.path.join(progress_dir, 'assd.out')       , assdi )
        # np.savetxt(os.path.join(progress_dir, 'asd.out')        , asdi  )
        np.savetxt(os.path.join(progress_dir, 'volc.out')       , volc  )
        np.savetxt(os.path.join(progress_dir, 'volcc.out')      , volcc )

        print('Files saved in', progress_dir)
        print('                            ')
        
    dice_mean = np.mean(dice  )
    dice_std  = np.std(dice   )
    jacc_mean = np.mean(jacc  )
    jacc_std  = np.std(jacc   )
    # prec_mean = np.mean(prec  )
    # prec_std  = np.std(prec   )
    reca_mean = np.mean(reca  )
    reca_std  = np.std(reca   ) 
    ravd_mean = np.mean(ravdi )
    ravd_std  = np.std(ravdi  )
    # hd_mean = np.mean(haud    )
    # hd_std  = np.std(haud     )
    # hd95_mean = np.mean(haud95)
    # hd95_std  = np.std(haud95 ) 
    assd_mean = np.mean(assdi )
    assd_std  = np.std(assdi  )
    # asd_mean = np.mean(asdi   )
    # asd_std  = np.std(asdi    )
    volc_mean = np.mean(volc  )
    volc_std  = np.std(volc   )
    volcc_mean = np.mean(volcc)
    volcc_std  = np.std(volcc )

    np.savetxt(os.path.join(progress_dir, 'dice_mean_std.out')       , [dice_mean, dice_std]  )
    np.savetxt(os.path.join(progress_dir, 'jaccard_mean_std.out')    , [jacc_mean, jacc_std]  )
    # np.savetxt(os.path.join(progress_dir, 'precision_mean_std.out')  , [prec_mean, prec_std]  )
    np.savetxt(os.path.join(progress_dir, 'recall_mean_std.out')     , [reca_mean, reca_std]  )
    np.savetxt(os.path.join(progress_dir, 'ravd_mean_std.out')       , [ravd_mean, ravd_std]  )
    # np.savetxt(os.path.join(progress_dir, 'hausdorff_mean_std.out')  , [hd_mean, hd_std]      )
    # np.savetxt(os.path.join(progress_dir, 'hausdorff95_mean_std.out'), [hd95_mean, hd95_std]  )
    np.savetxt(os.path.join(progress_dir, 'assd_mean_std.out')       , [assd_mean, assd_std]  )
    # np.savetxt(os.path.join(progress_dir, 'asd_mean_std.out')        , [asd_mean, asd_std]    )
    np.savetxt(os.path.join(progress_dir, 'volc_mean_std.out')       , [volc_mean, volc_std]  )
    np.savetxt(os.path.join(progress_dir, 'volcc_mean_std.out')      , [volcc_mean, volcc_std])

    print('Results:                                                              ')
    print('Mean Dice                               :', '{:.6f}'.format(dice_mean) )
    print('Mean Jaccard                            :', '{:.6f}'.format(jacc_mean) )
    # print('Mean Precision                          :', '{:.6f}'.format(prec_mean) )
    print('Mean Recall                             :', '{:.6f}'.format(reca_mean) )
    print('Mean Relative absolute volume difference:', '{:.6f}'.format(ravd_mean) )
    # print('Mean Hausdorff distance                 :', '{:.6f}'.format(hd_mean)   )
    # print('Mean Hausdorff distance (95%)           :', '{:.6f}'.format(hd95_mean) )
    print('Mean Average symmetric surface distance :', '{:.6f}'.format(assd_mean) )
    # print('Mean Average surface distance           :', '{:.6f}'.format(asd_mean)  )
    print('Mean Volume correlation                 :', '{:.6f}'.format(volc_mean) ) 
    print('Mean Volume change correlation          :', '{:.6f}'.format(volcc_mean)) 

    print(f'Evaluation took {time() - start} s ({len(dice)} images)')
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

    voxel_spacing = nib.load(label).header['pixdim'][1:4]

    label = nib.load(label).get_data()
    pred  = nib.load(pred ).get_data()

    # Compute evaluation metrics
    print(f'Evaluation for {pat_id}')
    dice = Dice(label, pred)
    print('Dice                                      :', '{:.6f}'.format(dice))
    jacc = jc(pred, label)
    print('Jaccard                                   :', '{:.6f}'.format(jacc))
    prec = precision(pred, label)
    print('Precision                                 :', '{:.6f}'.format(prec))
    reca = recall(pred, label)
    print('Recall                                    :', '{:.6f}'.format(reca))
    haud = hd(pred, label, voxelspacing=voxel_spacing, connectivity=1)
    print('Hausdorff distance                        :', '{:.6f}'.format(haud))
    haud95 = hd95(pred, label, voxelspacing=voxel_spacing, connectivity=1)
    print('Hausdorff distance 95%                    :', '{:.6f}'.format(haud95))
    assdi, asdi = assd_asd(pred, label, voxelspacing=voxel_spacing, connectivity=1)
    print('Average symetric surface distance       :', '{:.6f}'.format(assdi))
    print('Average surface distance                  :', '{:.6f}'.format(asdi))
    ravdi = ravd(pred, label)
    print('Relative absolute volume difference       :', '{:.6f}'.format(ravdi))
    volc = volume_correlation(pred, label)
    print('Volume correlation                        :', '{:.6f}'.format(volc[0]))
    volcc = volume_change_correlation(pred, label)
    print('Volume change correlation                 :', '{:.6f}'.format(volcc[0]))
    print('                                                                      ')

    return [dice, jacc, prec, reca, haud, haud95, assdi, asdi, ravdi, volc[0], volcc[0]]


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


def distance_under_1_3_mm(pred, label, voxel_size):
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
    # dt2 = distance_transform_edt(~pred, sampling=voxel_size)

    # fn = np.logical_and(label, np.logical_or(label, pred)) # Only fn voxels (check)

    dist_pred = dt[pred]
    # dist_fn = dt2[fn]

    dist_under_1 = []
    dist_under_3 = []

    for voxel in dist_pred:
        if voxel < 1:
            np.append(dist_under_1, voxel)
        if voxel < 3:
            np.append(dist_under_1, voxel)
            np.append(dist_under_3, voxel)

    per_under_1mm = len(dist_under_1) / dist_pred
    per_under_3mm = len(dist_under_3) / dist_pred

    return  per_under_1mm, per_under_3mm, dist_pred, dist_under_1, dist_under_3

##############################################################################################
#--------------------------------------------------------------------------------------------#    
##############################################################################################

def assd_asd(result, reference, voxelspacing=None, connectivity=1):
    """
    Average symmetric surface distance and Average surface distance.

    From medpy.metric.binary
    
    """
    asdi = asd(result, reference, voxelspacing, connectivity)
    assd = np.mean( (asdi, asd(reference, result, voxelspacing, connectivity)) )
    return assd, asdi