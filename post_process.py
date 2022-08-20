import pandas as pd
import SimpleITK as sitk
import numpy as np
from utils import extract_dataset, create_path, read_img_sitk
import os


def replace_enhancing_tumor(numpy_label):
    """
    In this function, we replace enhancing tumor with necrosis when the volume of predicted enhancing tumor is less than the threshold
    """
    ncr = numpy_label == 1
    ncr = np.array(ncr, dtype=np.float32)
    ed = numpy_label == 2
    ed = np.array(ed, dtype=np.float32)
    et = numpy_label == 4
    et = np.array(et, dtype=np.float32)
    mask = np.zeros_like(et)
    mask[et > 0] = 1  # replacing enhancing tumor with necrosis based on our threshold
    mask[ncr > 0] = 1
    mask[ed > 0] = 2
    return mask


def main(config_file):
    dataset_path = config_file['path']['dataset']
    extract_dataset(dataset_path, 'Validation')
    data_paths = create_path('./Validation', val=True)
    # read results provided by model
    p_results = pd.read_csv("./Stats_Validation_final.csv")
    ids = p_results.loc[:65, "Label"]

    et_zero_idices = []  # patients whose dice_et is 0 (false positive)
    for i, patient_id in enumerate(ids):
        if p_results.loc[i, 'Dice_ET'] == 0:
            et_zero_idices.append(i)

    et_volumes = []
    for i, patient_id in enumerate(ids):
        label = read_img_sitk(data_paths[i]['seg'])
        np_label = sitk.GetArrayFromImage(label)
        et = np_label == 4
        et = np.array(et, dtype=np.float32)
        et_volume = np.sum(et)
        et_volumes.append(et_volume)

    replace_et = []
    for i in et_zero_idices:
        if 40 >= et_volumes[i]:
            replace_et.append(ids[i])

    parent = '/content/submissions/Postprocessed'
    os.makedirs(parent)

    for i, patient_id in enumerate(ids):
        label = read_img_sitk(data_paths[i]['seg'])
        if patient_id in replace_et:
            np_label = sitk.GetArrayFromImage(label)
            new_np_label = replace_enhancing_tumor(np_label)
            new_label = sitk.GetImageFromArray(new_np_label)
        else:
            new_label = label

        path = parent + '/' + patient_id + '.nii.gz'
        sitk.WriteImage(new_label, path)
