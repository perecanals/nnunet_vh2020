import numpy as np

import matplotlib.pyplot as plt

import torch

from monai.transforms import ScaleIntensity
from monai.utils.misc import ensure_tuple

################################################################################################
#----------------------------------------------------------------------------------------------# 

def shuffle_order(x, seed=0):
    ''' Returns shuffled array of order indices

        Inputs:
            - x: array or list (1D) to be ordered.
            - seed: random seed

        Returns:
            - order: np array [len(x)].

    '''

    np.random.seed(seed)
    order = np.arange(len(x))
    np.random.shuffle(order)

    return order


################################################################################################
#----------------------------------------------------------------------------------------------#    

def timer(time_to_compute):
    ''' Silly function to display time in a visual way '''

    hrs = int(time_to_compute // 3600)
    mins = int((time_to_compute - 3600 * hrs) // 60)
    secs = time_to_compute % 60 
    
    return '{}h {}min {:.2f}s'.format(hrs, mins, secs)


################################################################################################
#----------------------------------------------------------------------------------------------#    

def show_slices(slices):
   ''' Function to display row of image slices '''

   _, axes = plt.subplots(1, len(slices), figsize=(16,9))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")


################################################################################################
#----------------------------------------------------------------------------------------------#    

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset.transform.set_random_state(worker_info.seed % (2 ** 32))


################################################################################################