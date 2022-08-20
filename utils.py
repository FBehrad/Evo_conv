import zipfile
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tensorflow.keras.callbacks import ModelCheckpoint
import glob
import re
from tensorflow import keras, math
import os


def extract_dataset(addr, target):
    dataset_path = addr
    zfile = zipfile.ZipFile(dataset_path)
    zfile.extractall(target)


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
    pattern = address + '/.*_(\w*)\.nii\.gz'
    pattern = re.compile(pattern)
    if not val:
        data_paths = [{pattern.findall(item)[0]: item for item in items} for items in
                      list(zip(t1, t2, t1ce, flair, seg))]
    else:
        data_paths = [{pattern.findall(item)[0]: item for item in items} for items in list(zip(t1, t2, t1ce, flair))]
    return data_paths


def create_destination(val, val_path, train_path, data_paths):
    if val:
        parent = val_path
    else:
        parent = train_path

    for i in range(len(data_paths)):
        path = data_paths[i]['t1'].split("/")
        s = path[3:-1]
        listToStr = '/'.join([str(elem) for elem in s])
        new_parent = os.path.join(parent, listToStr)
        os.makedirs(new_parent)

        
def create_path_os(address, ids, val=False):
    data_paths = []
    if not val:
        add = address + '/*/*/*GG/'
    else:
        add = address + '/*/*/'

    for i, patient_id in enumerate(ids):
        data_path = {}
        t1_add = glob.glob(add + patient_id + '/*t1.nii.gz')[0]
        t2_add = glob.glob(add + patient_id + '/*t2.nii.gz')[0]
        t1ce_add = glob.glob(add + patient_id + '/*t1ce.nii.gz')[0]
        flair_add = glob.glob(add + patient_id + '/*flair.nii.gz')[0]
        seg_add = glob.glob(add + patient_id + '/*seg.nii.gz')[0]
        data_path['flair'] = flair_add
        data_path['t1'] = t1_add
        data_path['t2'] = t2_add
        data_path['t1ce'] = t1ce_add
        data_path['seg'] = seg_add
        data_paths.append(data_path)

    return data_paths


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
    # print('New lr : ', new_lr)
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


