import glob


def create_path(address, ids, submission=None, train=False, val=False):
    data_paths = []
    if train:
        add = address + '/*GG/'
    elif val:
        add = address + '/'
        seg = submission + '/'

    for i, patient_id in enumerate(ids):

        data_path = {}
        t1_add = glob.glob(add + patient_id + '/*t1.nii.gz')[0]
        t2_add = glob.glob(add + patient_id + '/*t2.nii.gz')[0]
        t1ce_add = glob.glob(add + patient_id + '/*t1ce.nii.gz')[0]
        flair_add = glob.glob(add + patient_id + '/*flair.nii.gz')[0]
        if train:
            seg_add = glob.glob(add + patient_id + '/*seg.nii.gz')[0]
        elif val:
            seg_add = glob.glob(seg + patient_id + '.nii.gz')[0]
        data_path['flair'] = flair_add
        data_path['t1'] = t1_add
        data_path['t2'] = t2_add
        data_path['t1ce'] = t1ce_add
        data_path['seg'] = seg_add
        data_paths.append(data_path)

    return data_paths
