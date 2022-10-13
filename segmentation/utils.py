import zipfile
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tensorflow.keras.callbacks import ModelCheckpoint
import glob
import re
from tensorflow import keras
from tensorflow import math
import os
import tensorflow as tf


def extract_dataset(addr, target):
    zfile = zipfile.ZipFile(addr)
    zfile.extractall('../')
    os.rename(addr[:-4], target)


def create_path(address, train=False, aug=False, val=False):
    if train:
        t1_pattern = address + '/*GG/*/*t1.nii.gz'
        t2_patern = address + '/*GG/*/*t2.nii.gz'
        flair_pattern = address + '/*GG/*/*flair.nii.gz'
        t1ce_pattern = address + '/*GG/*/*t1ce.nii.gz'
        seg_pattern = address + '/*GG/*/*seg.nii.gz'  # Ground truth
    elif aug:
        t1_pattern = address + '/*GG/*/*/*t1.nii.gz'
        t2_patern = address + '/*GG/*/*/*t2.nii.gz'
        flair_pattern = address + '/*GG/*/*/*flair.nii.gz'
        t1ce_pattern = address + '/*GG/*/*/*t1ce.nii.gz'
        seg_pattern = address + '/*GG/*/*/*seg.nii.gz'
    elif val:
        t1_pattern = address + '/*/*t1.nii.gz'
        t2_patern = address + '/*/*t2.nii.gz'
        flair_pattern = address + '/*/*flair.nii.gz'
        t1ce_pattern = address + '/*/*t1ce.nii.gz'
    t1 = glob.glob(t1_pattern)
    t2 = glob.glob(t2_patern)
    flair = glob.glob(flair_pattern)
    t1ce = glob.glob(t1ce_pattern)
    if not val:
        seg = glob.glob(seg_pattern)
    pattern = re.compile('.*_(\w*)\.nii\.gz')
    if not val:
        data_paths = [{pattern.findall(item)[0]: item for item in items} for items in
                      list(zip(t1, t2, t1ce, flair, seg))]
    else:
        data_paths = [{pattern.findall(item)[0]: item for item in items} for items in list(zip(t1, t2, t1ce, flair))]
    return data_paths


def create_destination(data_paths, val=False, train=False, os_train=False, os_val=False):
    if val:
        parent = '../val_preprcessed_data'

    if train:
        parent = '../preprocessed_data'

    if os_train:
        parent = '../preprocessed_os_data'

    if os_val:
        parent = '../val_preprocessed_os_data'
    for i in range(len(data_paths)):
        path = data_paths[i]['t1'].split("\\")
        if val or os_val:
            intermediate = path[-2]
            new_parent = os.path.join(parent, intermediate)
        if train or os_train:
            intermediate = path[-3:-1]
            listToStr = '/'.join([str(elem) for elem in intermediate])
            new_parent = os.path.join(parent, listToStr)
        os.makedirs(new_parent)


def create_batch_of_path(data_paths):
    labels = []
    paths = []
    for i, data_path in enumerate(data_paths):
        path = []
        path.append(data_path['flair'])
        path.append(data_path['t1'])
        path.append(data_path['t2'])
        path.append(data_path['t1ce'])
        label = data_path['seg']
        paths.append(path)
        labels.append(label)

    return paths, labels


def read_img_nii(img_path):
    image_data = np.array(nib.load(img_path).get_fdata())
    return image_data


def read_img_sitk(img_path):
    image_data = sitk.ReadImage(img_path)
    return image_data


def save_best_model(path, monitor='val_dice_coefficient', mode='max'):
    checkpoit_best = ModelCheckpoint(path,
                                     save_weights_only=True,
                                     save_best_only=True,
                                     monitor=monitor,
                                     mode=mode,
                                     verbose=1)
    return checkpoit_best


def scheduler(epoch, lr):
    epochs = 300
    decay = (1 - ((epoch) / epochs)) ** 0.9
    new_lr = lr * decay
    return new_lr


def dice_coefficient(y_true, y_pred):
    intersection = math.reduce_sum(math.abs(y_true * y_pred), axis=[-3, -2, -1])
    first_sum = math.reduce_sum(math.square(y_true), axis=[-3, -2, -1])
    second_sum = math.reduce_sum(math.square(y_pred), axis=[-3, -2, -1])
    dn = math.add(first_sum, second_sum)
    epsilon = 1e-8
    f_dn = math.add(dn, epsilon)
    dice = math.reduce_mean(2 * intersection / f_dn)
    return dice


def loss_gt(e=1e-8):
    def loss_gt_(y_true, y_pred):
        intersection = math.reduce_sum(math.abs(y_true * y_pred), axis=[-3, -2, -1])
        first_sum = math.reduce_sum(math.square(y_true), axis=[-3, -2, -1])
        second_sum = math.reduce_sum(math.square(y_pred), axis=[-3, -2, -1])
        dn = math.add(first_sum, second_sum)
        epsilon = 1e-8
        f_dn = math.add(dn, epsilon)
        dice = math.reduce_mean(2 * intersection / f_dn)
        return 1 - dice

    return loss_gt_


class my_custom_generator_segmentation(keras.utils.Sequence):

    # custom generator to read a batch of data
    def __init__(self, data_paths, label_paths, batch_size):
        self.data_paths = data_paths
        self.label_paths = label_paths
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.data_paths) / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):

        batch_x = self.data_paths[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.label_paths[idx * self.batch_size: (idx + 1) * self.batch_size]

        items = []
        for item in batch_x:
            flair_img = read_img_nii(item[0])
            t1_img = read_img_nii(item[1])
            t2_img = read_img_nii(item[2])
            t1ce_img = read_img_nii(item[3])
            data = np.array([t1_img, t2_img, t1ce_img, flair_img], dtype=np.float32)
            items.append(data)

        labels = []
        for label_address in batch_y:
            label = read_img_nii(label_address)
            ncr = label == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET)
            ed = label == 2  # Peritumoral Edema (ED)
            et = label == 4  # GD-enhancing Tumor (ET)
            y = np.array([ncr, ed, et], dtype=np.float32)
            labels.append(y)

        return np.array(items), np.array(labels)


def get_flops(path):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(path)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)

            return flops.total_float_ops

