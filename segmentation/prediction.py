import os
import numpy as np
from utils import read_img_nii, create_path
from model import build_model
import SimpleITK as sitk
import yaml


def predict(item_path, model):
    flair_img = read_img_nii(item_path['flair'])
    t1_img = read_img_nii(item_path['t1'])
    t2_img = read_img_nii(item_path['t2'])
    t1ce_img = read_img_nii(item_path['t1ce'])
    data = np.array([t1_img, t2_img, t1ce_img, flair_img], dtype=np.float32)
    data = data[np.newaxis, ...]
    prediction = model.predict(data)
    prediction = np.where(prediction >= 0.7, 1, 0)  # the output of sigmoid is probabilities
    prediction = prediction[0, :, :, :, :]
    ncr = prediction[0, :, :, :]
    ed = prediction[1, :, :, :]
    et = prediction[2, :, :, :]
    ed = np.where(ed == 1, 2, ed)
    et = np.where(et == 1, 4, et)
    final_mask = np.maximum(ncr, ed)
    final_mask = np.maximum(final_mask, et)
    final_mask = final_mask[:, :, 5:]  # To remove padding that we perform in preprocessing
    return final_mask


if __name__ == '__main__':
    path = open('../config.yaml', 'r')
    config = yaml.safe_load(path)
    model_param = config['model']
    parent = '../submissions'
    os.makedirs(parent)

    # for validation
    model = build_model(input_shape=(4, 240, 240, 160),
                        gradient_accumulation=model_param['accumulated_grad']['enable'],
                        n_gradients=model_param['accumulated_grad']['num_batch'])
    model.load_weights(config['path']['best_model']).expect_partial()
    data_paths = create_path('../val_preprcessed_data', val=True)
    for i in range(len(data_paths)):
        item = data_paths[i]
        final_mask = predict(item, model)
        j = i + 1
        print("Mask " + str(j) + " is predicted")
        # Saving mask
        path = data_paths[i]['flair'].split('\\')
        patient_id = path[-2] + '.nii.gz'
        final_path = os.path.join(parent, patient_id)
        final_mask = np.moveaxis(final_mask, -1, 0)  # necessary for Conversion between numpy and SimpleITK
        final_mask = np.moveaxis(final_mask, -1, 1)
        sitk_mask = sitk.GetImageFromArray(final_mask)
        sitk.WriteImage(sitk_mask, final_path)


