import os
import shutil

src = "/Users/pere/Documents/Projects/Vall d'Hebron/Tortuositat/Database/Database Pere/NIFTIS"
dst_im = "/Users/pere/opt/anaconda3/envs/nnunet_env/nnUNet/nnunet/database_vh/database_images"
dst_la = "/Users/pere/opt/anaconda3/envs/nnunet_env/nnUNet/nnunet/database_vh/database_labels"

for idx in range(1,8):
    src_idx = os.path.join(src, f'batch0{idx}')
    batch = os.listdir(src_idx)
    for patient in batch:
        if len(patient) == 8:
            shutil.copyfile(os.path.join(src_idx, patient, f'{patient}_CTA.nii.gz'), os.path.join(dst_im, f'{patient}.nii.gz'))
            shutil.copyfile(os.path.join(src_idx, patient, f'{patient}_label.nii.gz'), os.path.join(dst_la, f'{patient}.nii.gz'))