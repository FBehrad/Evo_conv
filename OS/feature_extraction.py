from radiomics import featureextractor
import SimpleITK as sitk
import numpy as np
import six
import skimage
from utils import extract_dataset, read_img_sitk
from skimage.feature import hog
import pandas as pd


def fractal_dimension(Z, threshold=0.9):
    def boxcount(Z, k):
        S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                            np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where((S > 0) & (S < k * k))[0])

    Z = (Z < threshold)
    p = min(Z.shape)
    n = 2 ** np.floor(np.log(p) / np.log(2))
    n = int(np.log(n) / np.log(2))
    sizes = 2 ** np.arange(n, 1, -1)
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


def histogram_intensity(np_label):
    histo = 0
    for i in range(np_label.shape[0]):
        fd, hog_image = hog(np_label[i, :, :], orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                            visualize=True, multichannel=False)
        histo = histo + np.sum((hog_image))
    histo = histo / 1000
    return histo


def feature_extractor(label_subregion, datapaths, extractor, val):
    x_diff = []
    y_diff = []
    z_diff = []
    x_centre = []
    y_centre = []
    z_centre = []
    ratio = []
    kurtosis = []
    entropy = []
    fractaldim = []
    histo = []
    minor_axis_length = []
    major_axis_length = []
    for i in range(len(datapaths)):
        img = read_img_sitk(datapaths[i]['t1'])
        np_img = sitk.GetArrayFromImage(img)
        label = read_img_sitk(datapaths[i]['seg'])
        label.SetOrigin(img.GetOrigin())
        np_label = sitk.GetArrayFromImage(label)
        sub_region = np_label == label_subregion
        sub_region = np.array(sub_region, dtype=np.int16)
        if val:
            sub_region = np.pad(sub_region, ((5, 0), (0, 0), (0, 0)), 'constant', constant_values=0)
        if np.all(sub_region == 0):
            ratio.append(0)
            x_centre.append(0)
            y_centre.append(0)
            z_centre.append(0)
            x_diff.append(120)
            y_diff.append(120)
            z_diff.append(82.5)
            kurtosis.append(0)
            entropy.append(0)
            fractaldim.append(0)
            histo.append(0)
            minor_axis_length.append(0)
            major_axis_length.append(0)
        else:
            sub_region_sitk = sitk.GetImageFromArray(sub_region)
            sub_region_result = extractor.execute(img, sub_region_sitk)
            ratio = np.sum(sub_region) / np.count_nonzero(np_img)
            ratio.append(ratio)
            FractalDim = fractal_dimension(sub_region)
            fractaldim.append(FractalDim)
            histo = histogram_intensity(sub_region)
            histo.append(histo)
            kurt = (kurtosis(kurtosis(kurtosis(sub_region, fisher=True))))
            kurtosis.append(kurt)
            entropy = skimage.measure.shannon_entropy(sub_region, base=2)
            entropy.append(entropy)
            for key, value in six.iteritems(sub_region_result):
                if key == 'diagnostics_Mask-original_CenterOfMassIndex':
                    x_centre.append(value[0])
                    y_centre.append(value[1])
                    z_centre.append(value[2])
                    x = abs(value[0] - 120)
                    y = abs(value[1] - 120)
                    z = abs(value[2] - 82.5)  # because we have added 5 zero-paddings to the begining of this channel
                    x_diff.append(x)
                    y_diff.append(y)
                    z_diff.append(z)
                if key == 'original_shape_MinorAxisLength':
                    minor_axis_length.append(value)
                if key == 'original_shape_MajorAxisLength':
                    major_axis_length.append(value)
    sub_region_data = [x_centre, y_centre, z_centre, x_diff, y_diff, z_diff, ratio, kurtosis, entropy, fractaldim,
                       histo, minor_axis_length, major_axis_length]
    return sub_region_data


def main(config_file):
    dataset_path = config_file['path']['os_mri_data']
    val_path = config_file['path']['os_val_mri_data']
    os_data = config_file['path']['os_data']
    os_val = config_file['path']['os_val_data']
    result_path = config_file['path']['os_features']
    result_val_path = config_file['path']['os_val_features']

    extract_dataset(dataset_path, 'Training')
    extract_dataset(val_path, 'Validation')

    survival_data = pd.read_csv(os_data)
    survival_val = pd.read_csv(os_val)

    ids = survival_data.loc[:, "BraTS18ID"]
    ids_val = survival_val.loc[:, 'BraTS18ID']

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(shape=['MinorAxisLength', 'MajorAxisLength'])

    et_info = feature_extractor(4, dataset_path, extractor, val=False)
    wt_info = feature_extractor(2, dataset_path, extractor, val=False)
    ncr_info = feature_extractor(1, dataset_path, extractor, val=False)
    all_info = et_info + wt_info + ncr_info
    survival_info = pd.DataFrame(all_info,
                                 columns=['ET_x_centre', 'ET_y_centre', 'ET_z_centre', 'ET_x_Diff', 'ET_y_Diff',
                                          'ET_z_Diff', 'ET_ratio', 'ET_kurtosis', 'ET_entropy', 'ET_fractaldim',
                                          'ET_histo', 'ET_MinorAxisLength', 'ET_MajorAxisLength', 'WT_x_centre',
                                          'WT_y_centre', 'WT_z_centre', 'WT_x_Diff', 'WT_y_Diff',
                                          'WT_z_Diff', 'WT_ratio', 'WT_kurtosis', 'WT_entropy', 'WT_fractaldim',
                                          'WT_histo', 'WT_MinorAxisLength', 'WT_MajorAxisLength', 'NCR_x_centre',
                                          'NCR_y_centre', 'NCR_z_centre', 'NCR_x_Diff', 'NCR_y_Diff',
                                          'NCR_z_Diff', 'NCR_ratio', 'NCR_kurtosis', 'NCR_entropy', 'NCR_fractaldim',
                                          'NCR_histo', 'NCR_MinorAxisLength', 'NCR_MajorAxisLength'])
    survival_info['BraTS18ID'] = ids
    survival_info['Age'] = survival_data['Age']
    survival_info['ResectionStatus'] = survival_data['ResectionStatus']
    survival_info['Survival'] = survival_data['Survival']

    et_val_info = feature_extractor(4, val_path, extractor, val=True)
    wt_val_info = feature_extractor(2, val_path, extractor, val=True)
    ncr_val_info = feature_extractor(1, val_path, extractor, val=True)
    all_val_info = et_val_info + wt_val_info + ncr_val_info
    survival_val_info = pd.DataFrame(all_val_info,
                                     columns=['ET_x_centre', 'ET_y_centre', 'ET_z_centre', 'ET_x_Diff', 'ET_y_Diff',
                                              'ET_z_Diff', 'ET_ratio', 'ET_kurtosis', 'ET_entropy', 'ET_fractaldim',
                                              'ET_histo', 'ET_MinorAxisLength', 'ET_MajorAxisLength', 'WT_x_centre',
                                              'WT_y_centre', 'WT_z_centre', 'WT_x_Diff', 'WT_y_Diff',
                                              'WT_z_Diff', 'WT_ratio', 'WT_kurtosis', 'WT_entropy', 'WT_fractaldim',
                                              'WT_histo', 'WT_MinorAxisLength', 'WT_MajorAxisLength', 'NCR_x_centre',
                                              'NCR_y_centre', 'NCR_z_centre', 'NCR_x_Diff', 'NCR_y_Diff',
                                              'NCR_z_Diff', 'NCR_ratio', 'NCR_kurtosis', 'NCR_entropy', 'NCR_fractaldim',
                                              'NCR_histo', 'NCR_MinorAxisLength', 'NCR_MajorAxisLength'])
    survival_val_info['BraTS18ID'] = ids_val
    survival_val_info['Age'] = survival_val['Age']
    survival_val_info['ResectionStatus'] = survival_val['ResectionStatus']

    survival_data.to_csv(result_path)
    survival_val.to_csv(result_val_path)
    
    
if __name__ == '__main__':
    main()
