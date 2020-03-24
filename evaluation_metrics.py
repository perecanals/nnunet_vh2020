import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms

import os
import nibabel as nib
from nibabel.testing import data_path

import sklearn
from sklearn.metrics import confusion_matrix

import time 

def eval_metrics():

    # nnunet_dir = "/Users/pere/opt/anaconda3/envs/nnunet_env/nnUNet/nnunet"
    nnunet_dir = "/home/perecanals/nnunet_env/nnUNet/nnunet"

    # start = time.time()

    path_label = join(nnunet_dir, "inference_test/labels")
    path_out = join(nnunet_dir, "inference_test/output")

    dir_label = os.fsencode(path_label)
    dir_out = os.fsencode(path_out)

    list_label = []; list_out = []

    for file in os.listdir(dir_label):
        filename = os.fsdecode(file)
        if filename.endswith(".gz"):
            list_label.append(filename)
            continue
        else:
            continue

    for file in os.listdir(dir_out):
        filename = os.fsdecode(file)
        if filename.endswith(".gz"):
            list_out.append(filename)
            continue
        else:
            continue

    list_label.sort()
    list_out.sort()

    print(list_label)
    print(list_out)

    accuracy = []; sensitivity = []; specificity = []; dice_score = []

    # print("time 1", time.time()-start, "s")

    for i in range(len(list_label)):

        start_loop = time.time()

        path_file_label = os.path.join(path_label, list_label[i])
        img_label = nib.load(path_file_label)
        path_file_out = os.path.join(path_out, list_out[i])
        img_out = nib.load(path_file_out)

        img_label_data = img_label.get_data()
        t_label = torch.tensor(img_label_data)
        img_out_data = img_out.get_data()
        t_out = torch.tensor(img_out_data)

        # y_label = t_label.numpy()
        # y_out = t_out.numpy()

        # dice = np.sum(y_out[y_label==1]==1)*2.0 / (np.sum(y_out==1) + np.sum(y_label==1))

        # dice_score.append(dice)

        # continue

        CM = confusion_matrix(t_label.view(-1), t_out.view(-1)) # Too slow!!!!!

        tn = CM[0,0] 
        fp = CM[0,1]
        fn = CM[1,0]
        tp = CM[1,1]

        acc = (tn + tp) / (tn + tp + fn + fp)
        sen = tp / (tp + fn)
        spe = tn / (tn + fp)
        dice = 2 * tp / (2 * tp + fp + fn)

        accuracy.append(acc)
        sensitivity.append(sen)
        specificity.append(spe)
        dice_score.append(dice)

        print('accuracy =', accuracy)
        print('sensitivity =', sensitivity)
        print('specificity =', specificity)
        print('dice score =', dice_score)

        # print("time 2", time.time()-start_loop, "s")

    acc_mean = np.mean(accuracy)
    acc_std = np.std(accuracy)

    sen_mean = np.mean(sensitivity)
    sen_std = np.std(sensitivity)

    spe_mean = np.mean(specificity)
    spe_std = np.std(specificity)

    dice_mean = np.mean(dice_score)
    dice_std = np.std(dice_score)

    # print('accuracy =', acc_mean, acc_std)
    # print('sensitivity =', sen_mean, sen_std)
    # print('specificity =', spe_mean, spe_std)
    # print('dice score =', dice_mean, dice_std)

    # print("time 3", time.time()-start, "s")

    eval_met = [acc_mean, acc_std, sen_mean, sen_std, spe_mean, spe_std, dice_mean, dice_std]

    return eval_met