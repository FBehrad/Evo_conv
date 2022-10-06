from utils import extract_dataset, read_img_sitk, create_path, create_destination
import numpy as np
from radiomics import featureextractor
import six
import SimpleITK as sitk
import os
import time
import yaml


def radiomics_features(input_image, input_mask):
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    result = extractor.execute(input_image, input_mask)
    centre_of_mass = []
    for key, value in six.iteritems(result):
        if key == 'diagnostics_Mask-original_CenterOfMassIndex':
            centre_of_mass.append(value[0])
            centre_of_mass.append(value[1])
            centre_of_mass.append(value[2])
    return centre_of_mass


def crop_images(centre_of_mass, optimal_roi, image_size, dim):
    if dim == 0:
        opt_roi = optimal_roi[0]
        centre = round(centre_of_mass[0])
        size = image_size[0]
    elif dim == 1:
        opt_roi = optimal_roi[1]
        centre = round(centre_of_mass[1])
        size = image_size[1]
    else:
        opt_roi = optimal_roi[2]
        centre = round(centre_of_mass[2])
        size = image_size[2]

    roi = round(opt_roi / 2)
    if centre - roi < 0:
        start = 0
        end = opt_roi
    elif centre + roi >= size:
        end = size - 1
        start = end - opt_roi
    else:
        start = centre - roi
        end = centre + roi

    return start, end


def rescale_image(image):
    resacleFilter = sitk.RescaleIntensityImageFilter()
    resacleFilter.SetOutputMaximum(255)
    resacleFilter.SetOutputMinimum(0)
    rescaled_img = resacleFilter.Execute(image)
    return rescaled_img


def mean_normalization(image):
    image = sitk.GetArrayFromImage(image)
    mask = np.where(image != 0)  # as we want to normalize images only based on non-zero voxels
    desired = image[mask]
    mean = np.mean(desired)
    std = np.std(desired)
    final_image = (image - mean) / std
    final_image = sitk.GetImageFromArray(final_image)
    return final_image


def max_min_normalization(image):
    image = sitk.GetArrayFromImage(image)
    max = np.max(image)
    min = np.min(image)
    final_image = (image - min) / (max - min)
    final_image = sitk.GetImageFromArray(final_image)
    return final_image


def n4_bias_correction(image):
    maskImage = sitk.OtsuThreshold(image, 0, 1, 200)
    image = sitk.Cast(image, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    output_corrected = corrector.Execute(image, maskImage)
    return output_corrected


def Adativehistogram_equalization(img, alpha, beta):
    # The parameter alpha controls how much the filter acts like the classical histogram equalization method (alpha = 0)
    # to how much the filter acts like an unsharp mask (alpha = 1)
    # The parameter beta controls how much the filter acts like an unsharp mask (beta = 0)
    # to much the filter acts like pass through (beta = 1, with alpha = 1)
    histogramEqualization = sitk.AdaptiveHistogramEqualizationImageFilter()
    histogramEqualization.SetAlpha(alpha)
    histogramEqualization.SetBeta(beta)
    equalized_volume = histogramEqualization.Execute(img)
    return equalized_volume


def preprocessing(data_paths, config, val):
    image_size = [240, 240, 155]
    optimal_roi = config['preprocessing_seg']['optimal_roi']
    start_time = time.time()
    for i in range(len(data_paths)):
        if not val:
            input_image = sitk.ReadImage(data_paths[i]['flair'])
            input_mask = sitk.ReadImage(data_paths[i]['seg'])
            centre_of_mass = radiomics_features(input_image, input_mask)
        for j, modal in enumerate(data_paths[i]):
            img = read_img_sitk(data_paths[i][modal])
            if modal != 'seg':
                if val:
                    parent = '../val_preprcessed_data'
                    array_form = sitk.GetArrayFromImage(img)
                    # The model input must be factors of 16
                    new_array_form = np.pad(array_form, ((5, 0), (0, 0), (0, 0)), 'constant', constant_values=0)
                    image = sitk.GetImageFromArray(new_array_form)
                else:
                    parent = '../preprocessed_data'
                    start_x, end_x = crop_images(centre_of_mass, optimal_roi, image_size, dim=0)
                    start_y, end_y = crop_images(centre_of_mass, optimal_roi, image_size, dim=1)
                    start_z, end_z = crop_images(centre_of_mass, optimal_roi, image_size, dim=2)
                    image = img[start_x:end_x, start_y:end_y, start_z:end_z]

                rescaled_img = rescale_image(image)

                if config['preprocessing_seg']['min_max']:
                    image = max_min_normalization(rescaled_img)
                else:
                    image = mean_normalization(rescaled_img)

                if config['preprocessing_seg']['bias_correction']:
                    image = n4_bias_correction(image)

                if config['preprocessing_seg']['adaptive']:
                    image = Adativehistogram_equalization(image, 1, 1)

                path = data_paths[i][modal].split('\\')
                if val:
                    intermediate_folders = path[-2:]
                else:
                    intermediate_folders = path[-3:]
                listToStr = '/'.join([str(elem) for elem in intermediate_folders])
                final_path = os.path.join(parent, listToStr)
                sitk.WriteImage(image, final_path)
            else:
                start_x, end_x = crop_images(centre_of_mass, optimal_roi, image_size, dim=0)
                start_y, end_y = crop_images(centre_of_mass, optimal_roi, image_size, dim=1)
                start_z, end_z = crop_images(centre_of_mass, optimal_roi, image_size, dim=2)
                image = img[start_x:end_x, start_y:end_y, start_z:end_z]
                path = data_paths[i][modal].split('\\')
                if val:
                    intermediate_folders = path[-2:]
                else:
                    intermediate_folders = path[-3:]
                listToStr = '/'.join([str(elem) for elem in intermediate_folders])
                final_path = os.path.join(parent, listToStr)
                sitk.WriteImage(image, final_path)

    end_time = time.time() - start_time
    hours, rem = divmod(end_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Preprocessing time :")
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == '__main__':
    path = open('../config.yaml', 'r')
    config = yaml.safe_load(path)

    training_data = config['path']['training_set']
    validation_data = config['path']['validation_set']

    if not os.path.exists('../Training'):
        extract_dataset(training_data, '../Training')

    if not os.path.exists('../Validation'):
        extract_dataset(validation_data, '../Validation')

    data_paths = create_path('../Training', train=True)
    data_paths_val = create_path('../Validation', val=True)

    # Create directories to save results
    if not os.path.exists('../preprocessed_data'):
        create_destination(data_paths, val=False)  # create directories for training data
    if not os.path.exists('../val_preprcessed_data'):
        create_destination(data_paths_val, val=True) # create directories for validation data

    preprocessing(data_paths, config, val=False)
    preprocessing(data_paths_val, config, val=True)
