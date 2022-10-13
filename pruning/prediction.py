from segmentation.prediction import predict
import os
import SimpleITK as sitk
from segmentation.utils import create_path, loss_gt, dice_coefficient
import tensorflow as tf
import numpy as np
import yaml


if __name__ == '__main__':
    path = open('../config.yaml', 'r')
    config = yaml.safe_load(path)
    if config['genetic']['version'] == 'third':
        path = '../Pruned_a_third_model.h5'
    else:
        path = '../Pruned_a_forth_model.h5'
    pruned_model = tf.keras.models.load_model('../Pruned_a_forth_model.h5',
                                              custom_objects={"loss_gt_": loss_gt(),
                                                              'dice_coefficient': dice_coefficient})

    # Predict masks for validation
    val_data_paths = create_path('../val_preprcessed_data', val=True)
    parent = '../pruning_submission'

    for i in range(len(val_data_paths)):
        item = val_data_paths[i]
        final_mask = predict(item, pruned_model)
        j = i + 1
        print("Mask " + str(j) + " is predicted")
        # Saving mask
        path = val_data_paths[i]['flair'].split('\\')
        patient_id = path[-2] + '.nii.gz'
        final_path = os.path.join(parent, patient_id)
        final_mask = np.moveaxis(final_mask, -1, 0)  # necessary for Conversion between numpy and SimpleITK
        final_mask = np.moveaxis(final_mask, -1, 1)
        sitk_mask = sitk.GetImageFromArray(final_mask)
        sitk.WriteImage(sitk_mask, final_path)

