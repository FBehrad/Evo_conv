import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
import yaml
from segmentation.utils import read_img_sitk, create_destination
from segmentation.preprocessing import rescale_image, max_min_normalization, mean_normalization
from utils import create_path


def preprocessing(data_paths, config, train=False, val=False):
    if train:
        parent = '../preprocessed_os_data'
    else:
        parent = '../val_preprocessed_os_data'
    for i in range(len(data_paths)):
        print('Image ', str(i + 1))
        for j, modal in enumerate(data_paths[i]):
            if modal != 'seg':
                img = read_img_sitk(data_paths[i][modal])
                array_form = sitk.GetArrayFromImage(img)
                new_array_form = np.pad(array_form, ((5, 0), (0, 0), (0, 0)), 'constant',
                                        constant_values=0)  # The model input must be factors of 16
                new_img = sitk.GetImageFromArray(new_array_form)
                rescaled_img = rescale_image(new_img)
                if config['preprocessing_seg']['min_max']:
                    image = max_min_normalization(rescaled_img)
                else:
                    image = mean_normalization(rescaled_img)

                path = data_paths[i][modal].split('/')
                intermediate_folders = path[-2:]
                listToStr = '/'.join([str(elem) for elem in intermediate_folders])
                final_path = os.path.join(parent, listToStr)
                sitk.WriteImage(image, final_path)

            else:
                img = read_img_sitk(data_paths[i][modal])
                array_form = sitk.GetArrayFromImage(img)
                new_array_form = np.pad(array_form, ((5, 0), (0, 0), (0, 0)), 'constant',
                                        constant_values=0)  # The model input must be factors of 16
                new_img = sitk.GetImageFromArray(new_array_form)
                path = data_paths[i][modal].split('\\')
                intermediate_folders = path[-2:]
                listToStr = '/'.join([str(elem) for elem in intermediate_folders])
                final_path = os.path.join(parent, listToStr)
                sitk.WriteImage(new_img, final_path)


if __name__ == '__main__':
    path = open('../config.yaml', 'r')
    config = yaml.safe_load(path)

    survival_data = pd.read_csv('../Training/survival_data.csv')
    ids = survival_data.loc[:, 'BraTS18ID']

    survival_val = pd.read_csv('../Validation/survival_evaluation.csv')
    ids_val = survival_val.loc[:, 'BraTS18ID']

    data_paths = create_path('../Training', ids, train=True)
    data_paths_val = create_path('../Validation', ids_val, submission='../submissions', val=True)

    # Create directories to save results
    if not os.path.exists('../preprocessed_os_data'):
        create_destination(data_paths, os_train=True)  # create directories for training data
    if not os.path.exists('../val_preprocessed_os_data'):
        create_destination(data_paths_val, os_val=True)  # create directories for validation data

    preprocessing(data_paths, config, train=True)
    preprocessing(data_paths_val, config, val=True)
