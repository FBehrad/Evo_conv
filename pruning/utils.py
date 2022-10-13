import numpy as np
import math


def order_layers(model):
    # Order layers for pruning
    conv_layers_indexes = []
    layers_dic = {}
    unordered_layers = {}

    for i in range(len(model.layers)):
        if model.layers[i].__class__.__name__ == 'Conv3D':
            conv_layers_indexes.append(i)

    for i in conv_layers_indexes:  # i = 0 3 6
        weight = model.layers[i].get_weights()[0]
        filter_shape = weight.shape
        nb_of_fils = filter_shape[-1]
        weight_dict = {}
        # computation of l2 norm for a wise initialization
        for j in range(nb_of_fils):
            square = np.square(abs(weight[:, :, :, :, j]))
            sum = np.sum(square)
            l2 = math.sqrt(sum)
            # l1 = np.sum(abs(weight[:,:,:,:,j])) # change this because not all networks have this dimension
            weight_dict[j] = np.array(l2)

        weight_dict_sorted = sorted(weight_dict.items(), key=lambda kv: kv[1])
        layers_dic[i] = weight_dict_sorted

    for x in layers_dic:
        ordered_filters = layers_dic[x]
        lowest_value = ordered_filters[0][-1]  # ordered_filters[0] -->(28, 26.955196)
        unordered_layers[x] = lowest_value

    ordered_layers = sorted(unordered_layers.items(), key=lambda kv: kv[1])
    return ordered_layers, layers_dic


def creat_pair_path(data_paths):
    labels_val = []
    paths_val = []
    for i, data_path in enumerate(data_paths):
        path = []
        path.append(data_path['flair'])
        path.append(data_path['t1'])
        path.append(data_path['t2'])
        path.append(data_path['t1ce'])
        label = data_path['seg']
        paths_val.append(path)
        labels_val.append(label)
    return paths_val, labels_val
