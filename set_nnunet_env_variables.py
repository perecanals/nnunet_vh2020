import os

mainDir = os.path.abspath("")

if os.path.basename(os.path.abspath("")) == "nnunet":
    if not os.path.isdir(os.path.join(mainDir, "nnUNet_base")): os.mkdir(os.path.join(mainDir, "nnUNet_base"))
    if not os.path.isdir(os.path.join(mainDir, "nnUNet_base/nnUNet_preprocessed")): os.mkdir(os.path.join(mainDir, "nnUNet_base/nnUNet_preprocessed"))
    if not os.path.isdir(os.path.join(mainDir, "nnUNet_base/nnUNet_training_output_dir")): os.mkdir(os.path.join(mainDir, "nnUNet_base/nnUNet_training_output_dir"))

    os.environ["nnUNet_raw_data_base"] = os.path.join(mainDir, "nnUNet_base")
    os.environ["nnUNet_preprocessed" ] = os.path.join(mainDir, "nnUNet_base/nnUNet_preprocessed")
    os.environ["RESULTS_FOLDER"      ] = os.path.join(mainDir, "nnUNet_base/nnUNet_training_output_dir")

else:
    ValueError("NOthing was done. Make sure you are in the ./nnunet dir")