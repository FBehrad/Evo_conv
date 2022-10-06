import SimpleITK as sitk  # important package for medical imaging
import elasticdeform
import numpy as np
import os
from utils import read_img_sitk, read_img_nii, create_path
import yaml


def brightness_shift(X, gain, gamma):
    """
    Changing the brighness of a image using power-law gamma transformation.
    Gain and gamma are chosen randomly for each image channel.

    Gain chosen between [0.8 - 1.2]
    Gamma chosen between [0.8 - 1.2]

    new_im = gain * im^gamma
    """
    im_new = np.sign(X) * gain * (np.abs(X) ** gamma)
    return im_new


def brightness(data_paths_aug, i):
    # create directories
    path = data_paths_aug[i]['t1'].split('\\')
    parent = '../augmented_data'
    intermediate_folders = path[-3:-1]
    tech = 'brightness'
    listToStr = '/'.join([str(elem) for elem in intermediate_folders]) + '/' + tech
    parent = os.path.join(parent, listToStr)
    if not os.path.exists(parent):
        os.makedirs(parent)

    # create parameters for random brightness shift
    gain, gamma = (1.2 - 0.8) * np.random.random_sample(2, ) + 0.8
    for j, modal in enumerate(data_paths_aug[i]):
        if modal != 'seg':
            image = read_img_nii(data_paths_aug[i][modal])
            new_img = brightness_shift(image, gain, gamma)
            new_img = np.moveaxis(new_img, -1, 0)  # necessary for Conversion between numpy and SimpleITK
            new_img = np.moveaxis(new_img, -1, 1)
            new_img = sitk.GetImageFromArray(new_img)
            path = data_paths_aug[i][modal].split('\\')
            file_name = path[-1]
            final_path = os.path.join(parent, file_name)
            sitk.WriteImage(new_img, final_path)
        elif modal == 'seg':
            label = read_img_sitk(data_paths_aug[i][modal])
            path = data_paths_aug[i][modal].split('\\')
            file_name = path[-1]
            final_path = os.path.join(parent, file_name)
            sitk.WriteImage(label, final_path)


def noisy_gauss(img, mean, var):
    row, col, ch = img.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy_img = img + gauss
    return noisy_img


def elastic(X, y):
    """
    Elastic deformation on a image and its target
    """
    [Xel, yel] = elasticdeform.deform_random_grid([X, y], sigma=2, axis=[(1, 2, 3), (0, 1, 2)], order=[1, 0],
                                                  mode='constant')
    return Xel, yel


def elastic_deformation(data_paths_aug, i):
    # create directories
    path = data_paths_aug[i]['t1'].split('\\')
    parent = '../augmented_data'
    intermediate_folders = path[-3:-1]
    tech = 'elastic_deformation'
    listToStr = '/'.join([str(elem) for elem in intermediate_folders]) + '/' + tech
    parent = os.path.join(parent, listToStr)
    if not os.path.exists(parent):
        os.makedirs(parent)

    t1 = read_img_nii(data_paths_aug[i]['t1'])
    t2 = read_img_nii(data_paths_aug[i]['t2'])
    flair = read_img_nii(data_paths_aug[i]['flair'])
    t1ce = read_img_nii(data_paths_aug[i]['t1ce'])
    data = np.array([t1, t2, t1ce, flair], dtype=np.float32)
    y = read_img_nii(data_paths_aug[i]['seg'])
    new_data, new_y = elastic(data, y)
    patient_info = {
        't1': new_data[0],
        't2': new_data[1],
        't1ce': new_data[2],
        'flair': new_data[3],
        'seg': new_y
    }
    for modal in patient_info:
        modality = patient_info[modal]
        new_img = np.moveaxis(modality, -1, 0)  # necessary for Conversion between numpy and SimpleITK
        new_img = np.moveaxis(new_img, -1, 1)
        sitk_img = sitk.GetImageFromArray(new_img)
        path = data_paths_aug[i][modal].split('\\')
        file_name = path[-1]
        final_path = os.path.join(parent, file_name)
        sitk.WriteImage(sitk_img, final_path)


if __name__ == '__main__':
    data_paths_aug = create_path('../preprocessed_data', train=True)
    path = open('../config.yaml', 'r')
    config = yaml.safe_load(path)
    aug = config['data_augmentation']
    for i in range(len(data_paths_aug)):
        if aug['brightness']['enable']:
            br = np.random.rand()
            if br <= aug['brightness']['limit']:
                brightness(data_paths_aug, i)
        if aug['elastic_deform']['enable']:
            el = np.random.rand()
            if el <= aug['elastic_deform']['limit']:
                elastic_deformation(data_paths_aug, i)
