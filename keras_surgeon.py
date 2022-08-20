import collections.abc
from tensorflow.keras.activations import linear
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
import tensorflow_addons as tfa


def inbound_nodes(layer):
    return layer.inbound_nodes


def make_list_if_not(x):
    if isinstance(x, collections.abc.Sequence) and not isinstance(x, str):
        return x
    else:
        return [x]


def node_indices(node):
    return make_list_if_not(node.node_indices)


def inbound_layers(node):
    return make_list_if_not(node.inbound_layers)


def parent_nodes(node):
    try:
        return node.parent_nodes
    except AttributeError:
        return [inbound_nodes(inbound_layers(node)[i])[node_index]
                for i, node_index in enumerate(node_indices(node))]


class TensorKeys(list):
    def __init__(self, refs):
        super().__init__(refs)

    def __contains__(self, item):
        try:
            return super().__contains__(item.ref())
        except AttributeError:
            return super().__contains__(item.experimental_ref())


class TensorDict(dict):
    def __init__(self):
        super().__init__()
        # self.d = {}

    def __setitem__(self, key, value):
        try:
            super().__setitem__(key.ref(), value)
        except AttributeError:
            super().__setitem__(key.experimental_ref(), value)

    def __getitem__(self, item):
        try:
            return super().__getitem__(item.ref())
        except AttributeError:
            return super().__getitem__(item.experimental_ref())

    def keys(self):
        return TensorKeys(super().keys())


"""Utilities used across other modules."""


def clean_copy(model):
    """Returns a copy of the model without other model uses of its layers."""
    weights = model.get_weights()
    new_model = model.__class__.from_config(model.get_config())
    new_model.set_weights(weights)
    return new_model


def get_channels_attr(layer):
    layer_config = layer.get_config()
    if 'units' in layer_config.keys():
        channels_attr = 'units'
    elif 'filters' in layer_config.keys():
        channels_attr = 'filters'
    else:
        raise ValueError('This layer has not got any channels.')
    return channels_attr


def get_node_depth(model, node):
    """Get the depth of a node in a model.
    Arguments:
        model: Keras Model object
        node: Keras Node object
    Returns:
        The node depth as an integer. The model outputs are at depth 0.
    Raises:
        KeyError: if the node is not contained in the model.
    """
    for (depth, nodes_at_depth) in model._nodes_by_depth.items():
        if node in nodes_at_depth:
            return depth
    raise KeyError('The node is not contained in the model.')


def check_for_layer_reuse(model, layers=None):
    """Returns True if any layers are reused, False if not."""
    if layers is None:
        layers = model.layers
    return any([len(l.inbound_nodes) > 1 for l in layers])


def find_nodes_in_model(model, layer):
    """Find the indices of layer's inbound nodes which are in model"""
    model_nodes = get_model_nodes(model)
    node_indices = []
    for i, node in enumerate(layer.inbound_nodes):
        if node in model_nodes:
            node_indices.append(i)
    return node_indices


def check_nodes_in_model(model, nodes):
    """Check if nodes are in model"""
    model_nodes = get_model_nodes(model)
    nodes_in_model = [False] * len(nodes)
    for i, node in enumerate(nodes):
        if node in model_nodes:
            nodes_in_model[i] = True
    return nodes_in_model


def get_model_nodes(model):
    """Return all nodes in the model"""
    return [node for v in model._nodes_by_depth.values() for node in v]


def get_shallower_nodes(node):
    possible_nodes = node.outbound_layer.outbound_nodes
    next_nodes = []
    for n in possible_nodes:
        if node in parent_nodes(n):
            next_nodes.append(n)
    return next_nodes


def get_node_index(node):
    for i, n in enumerate(node.outbound_layer.inbound_nodes):
        if node == n:
            return i


def find_activation_layer(layer, node_index):
    """
    Args:
        layer(Layer):
        node_index:
    """
    output_shape = layer.get_output_shape_at(node_index)
    maybe_layer = layer
    node = maybe_layer.inbound_nodes[node_index]
    # Loop will be broken by an error if an output layer is encountered
    while True:
        # If maybe_layer has a nonlinear activation function return it and its index
        activation = getattr(maybe_layer, 'activation', linear)
        if activation.__name__ != 'linear':
            if maybe_layer.get_output_shape_at(node_index) != output_shape:
                ValueError('The activation layer ({0}), does not have the same'
                           ' output shape as {1}'.format(maybe_layer.name,
                                                         layer.name))
            return maybe_layer, node_index

        # If not, move to the next layer in the datastream
        next_nodes = get_shallower_nodes(node)
        # test if node is a list of nodes with more than one item
        if len(next_nodes) > 1:
            ValueError('The model must not branch between the chosen layer'
                       ' and the activation layer.')
        node = next_nodes[0]
        node_index = get_node_index(node)
        maybe_layer = node.outbound_layer

        # Check if maybe_layer has weights, no activation layer has been found
        if maybe_layer.weights and (
                not maybe_layer.__class__.__name__.startswith('Global')):
            AttributeError('There is no nonlinear activation layer between {0}'
                           ' and {1}'.format(layer.name, maybe_layer.name))


def sort_x_by_y(x, y):
    """Sort the iterable x by the order of iterable y"""
    x = [x for (_, x) in sorted(zip(y, x))]
    return x


def single_element(x):
    """If x contains a single element, return it; otherwise return x"""
    if isinstance(x, tf.Tensor):
        return x

    if len(x) == 1:
        x = x[0]
    return x


def get_one_tensor(x):
    if isinstance(x, tf.Tensor):
        return x

    assert len(x) == 1
    return x[0]


def bool_to_index(x):
    return [i for i, v in enumerate(x) if v]


def all_equal(iterator):
    try:
        iterator = iter(iterator)
        first = next(iterator)
        return all(
            np.array_equal(first, rest) for rest in iterator)
    except StopIteration:
        return True


class MeanCalculator:
    def __init__(self, sum_axis):
        self.values = None
        self.n = 0
        self.sum_axis = sum_axis

    def add(self, v):
        if self.values is None:
            self.values = v.sum(axis=self.sum_axis)
        else:
            self.values += v.sum(axis=self.sum_axis)
        self.n += v.shape[self.sum_axis]

    def calculate(self):
        return self.values / self.n


"""Identify which channels to delete."""


def get_apoz(model, layer, x_val, node_indices=None):
    """Identify neurons with high Average Percentage of Zeros (APoZ).
    The APoZ a.k.a. (A)verage (P)ercentage (o)f activations equal to (Z)ero,
    is a metric for the usefulness of a channel defined in this paper:
    "Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient
    Deep Architectures" - [Hu et al. (2016)][]
    `high_apoz()` enables the pruning methodology described in this paper to be
    replicated.
    If node_indices are not specified and the layer is shared within the model
    the APoZ will be calculated over all instances of the shared layer.
    Args:
        model: A Keras model.
        layer: The layer whose channels will be evaluated for pruning.
        x_val: The input of the validation set. This will be used to calculate
            the activations of the layer of interest.
        node_indices(list[int]): (optional) A list of node indices.
    Returns:
        List of the APoZ values for each channel in the layer.
    """

    if isinstance(layer, str):
        layer = model.get_layer(name=layer)

    # Check that layer is in the model
    if layer not in model.layers:
        raise ValueError('layer is not a valid Layer in model.')

    layer_node_indices = find_nodes_in_model(model, layer)
    # If no nodes are specified, all of the layer's inbound nodes which are
    # in model are selected.
    if not node_indices:
        node_indices = layer_node_indices
    # Check for duplicate node indices
    elif len(node_indices) != len(set(node_indices)):
        raise ValueError('`node_indices` contains duplicate values.')
    # Check that all of the selected nodes are in the layer
    elif not set(node_indices).issubset(layer_node_indices):
        raise ValueError('One or more nodes specified by `layer` and '
                         '`node_indices` are not in `model`.')

    data_format = getattr(layer, 'data_format', 'channels_last')
    # Perform the forward pass and get the activations of the layer.
    mean_calculator = MeanCalculator(sum_axis=0)
    for node_index in node_indices:
        act_layer, act_index = find_activation_layer(layer, node_index)
        # Get activations
        temp_model = Model(model.inputs, act_layer.get_output_at(act_index))
        a = temp_model.predict(x_val)

        if data_format == 'channels_first':
            a = np.swapaxes(a, 1, -1)
        # Flatten all except channels axis
        activations = np.reshape(a, [-1, a.shape[-1]])
        zeros = (activations == 0).astype(int)
        mean_calculator.add(zeros)

    return mean_calculator.calculate()


def high_apoz(apoz, method="std", cutoff_std=1, cutoff_absolute=0.99):
    """
    Args:
        apoz: List of the APoZ values for each channel in the layer.
        method: Cutoff method for high APoZ. "std", "absolute" or "both".
        cutoff_std: Channels with a higher APoZ than the layer mean plus
            `cutoff_std` standard deviations will be identified for pruning.
        cutoff_absolute: Channels with a higher APoZ than `cutoff_absolute`
            will be identified for pruning.
    Returns:
        high_apoz_channels: List of indices of channels with high APoZ.
    """
    if method not in {'std', 'absolute', 'both'}:
        raise ValueError('Invalid `mode` argument. '
                         'Expected one of {"std", "absolute", "both"} '
                         'but got', method)
    if method == "std":
        cutoff = apoz.mean() + apoz.std()*cutoff_std
    elif method == 'absolute':
        cutoff = cutoff_absolute
    else:
        cutoff = min([cutoff_absolute, apoz.mean() + apoz.std()*cutoff_std])

    cutoff = min(cutoff, 1)

    return np.where(apoz >= cutoff)[0]


# Set up logging
logging.basicConfig(level=logging.INFO)


class Surgeon:
    """Performs network surgery on a model.
    Surgeons can perform multiple network surgeries (jobs) at once.
    This is much faster than performing them sequentially.
    See `add_jobs` for a list of valid jobs and their required keyword arguments.
    Examples:
        Delete some channels from layer_1 and layer_2:
            surgeon = Surgeon(model)
            surgeon.add_job('delete_channels', layer_1, channels_1)
            surgeon.add_job('delete_channels', layer_2, channels_2)
            new_model = surgeon.operate()
    Arguments:
        model: The model to be modified
        copy: If True, the model will be copied before and after any operations
              This keeps the layers in the original model and the new model separate.
    """
    def __init__(self, model, copy=None):
        if copy:
            self.model = clean_copy(model)
        else:
            self.model = model
        self.nodes = []
        self._copy = copy
        self._finished_nodes = {}
        self._replace_tensors = TensorDict()
        self._channels_map = {}
        self._new_layers_map = {}
        self._insert_layers_map = {}
        self._replace_layers_map = {}
        self._mod_func_map = {}
        self._kwargs_map = {}
        self.valid_jobs = ('delete_layer',
                           'insert_layer',
                           'replace_layer',
                           'delete_channels')

    def add_job(self, job, layer, *,
                channels=None, new_layer=None, node_indices=None):
        """Adds a job for the Surgeon to perform on the model.
        Job options are:
        'delete_layer': delete `layer` from the model
                        required keyword arguments: None
        'insert_layer': insert `new_layer` before `layer`
                        required keyword arguments: `new_layer`
        'replace_layer': replace `layer` with `new_layer`
                         required keyword arguments: `new_layer`
        'delete_channels': delete `channels` from `layer`
                           required keyword arguments: `channels`
        Jobs can be added in any order. They will be performed in order of
        decreasing network depth.
        A maximum of one job can be performed per node.
        Args:
            job(string): job identifier. One of `Surgeon.valid_jobs`.
            layer(Layer): A layer from `model` to be modified.
            channels(list[int]): A list of channels used for the job.
                                 Used in `delete_channels`.
            new_layer(Layer): A new layer used for the job. Used in
                              `insert_layer` and `replace_layer`.
            node_indices(list[int]): (optional) A list of node indices used to
                                    selectively apply the job to a subset of
                                    the layer's nodes. Nodes are selected with:
                                    node[i] = layer.inbound_nodes[node_indices[i]]
        """
        # If the model has been copied, identify `layer` in the copied model.
        if self._copy:
            layer = self.model.get_layer(layer.name)
        # Check that layer is in the model
        if layer not in self.model.layers:
            raise ValueError('layer is not a valid Layer in model.')

        layer_node_indices = find_nodes_in_model(self.model, layer)
        # If no nodes are specified, all of the layer's inbound nodes which are
        # in model are selected.
        if not node_indices:
            node_indices = layer_node_indices
        # Check for duplicate node indices
        elif len(node_indices) != len(set(node_indices)):
            raise ValueError('`node_indices` contains duplicate values.')
        # Check that all of the selected nodes are in the layer
        elif not set(node_indices).issubset(layer_node_indices):
            raise ValueError('One or more nodes specified by `layer` and '
                             '`node_indices` are not in `model`.')

        # Select the modification function and any keyword arguments.
        kwargs = {}
        if job == 'delete_channels':
            # If not all inbound_nodes are selected, the new layer is renamed
            # to avoid duplicate layer names.
            if set(node_indices) != set(layer_node_indices):
                kwargs['layer_name'] = layer.name + '_' + job
            kwargs['channels'] = channels
            mod_func = self._delete_channels

        elif job == 'delete_layer':
            mod_func = self._delete_layer

        elif job == 'insert_layer':
            kwargs['new_layer'] = new_layer
            mod_func = self._insert_layer

        elif job == 'replace_layer':
            kwargs['new_layer'] = new_layer
            mod_func = self._replace_layer

        else:
            raise ValueError(job + ' is not a recognised job. Valid jobs '
                             'are:\n-', '\n- '.join(self.valid_jobs))

        # Get nodes to be operated on for this job
        job_nodes = []
        for node_index in node_indices:
            job_nodes.append(layer.inbound_nodes[node_index])
        # Check that the nodes do not already have jobs assigned to them.
        if set(job_nodes).intersection(self.nodes):
            raise ValueError('Cannot apply several jobs to the same node.')

        # Add the modification function and keyword arguments to the
        # self._mod_func_map and _kwargs_map dictionaries for later retrieval.
        for node in job_nodes:
            self._mod_func_map[node] = mod_func
            self._kwargs_map[node] = kwargs
        self.nodes.extend(job_nodes)

    def operate(self):
        """Perform all jobs assigned to the surgeon.
        """
        # Operate on each node in self.nodes by order of decreasing depth.
        sorted_nodes = sorted(self.nodes, reverse=True,
                              key=lambda x: get_node_depth(self.model, x))
        for node in sorted_nodes:
            # Rebuild submodel up to this node
            sub_output_nodes = parent_nodes(node)
            outputs, output_masks = self._rebuild_graph(self.model.inputs,
                                                        sub_output_nodes)

            # Perform surgery at this node
            kwargs = self._kwargs_map[node]
            self._mod_func_map[node](node, outputs, output_masks, **kwargs)

        # Finish rebuilding model
        output_nodes = []
        for output in self.model.outputs:
            layer, node_index, tensor_index = output._keras_history
            output_nodes.append(layer.inbound_nodes[node_index])
        new_outputs, _ = self._rebuild_graph(self.model.inputs, output_nodes)
        new_model = Model(self.model.inputs, new_outputs)

        if self._copy:
            return clean_copy(new_model)
        else:
            return new_model

    def _rebuild_graph(self,
                       graph_inputs,
                       output_nodes,
                       graph_input_masks=None):
        """Rebuild the graph from graph_inputs to output_nodes.
        This does not return a model object, it re-creates the connections
        between layers and returns the output tensors and masks of the submodel
        This is a building block for the higher level surgery methods.
        See `Surgeon.operate` for details of how this method is used.
        Arguments:
            graph_inputs: List of the submodel's input tensor(s).
            output_nodes(list[Node]): List of the submodel's output node(s)
            graph_input_masks: Boolean mask for each submodel input.
        Returns:
            (tuple) containing :
                List of the output tensors of the rebuilt submodel
                List of the output masks of the rebuilt submodel
            tuple[submodel output tensors, output masks]
        """
        if not graph_input_masks:
            graph_input_masks = [None] * len(graph_inputs)

        def _rebuild_rec(node):
            """Rebuild the graph up to `node` recursively.
            Args:
                node(Node): Node to rebuild up to.
            Returns:
                (tuple) containing :
                The output tensor of the rebuilt `node`
                The output mask of the rebuilt `node`
            """
            layer = node.outbound_layer
            logging.debug('getting inputs for: {0}'.format(layer.name))
            node_output = single_element(node.output_tensors)
            # First check for conditions to bottom out the recursion
            # Check for replaced tensors before any other checks:
            # these are created by the surgery methods.
            if node_output in self._replace_tensors.keys():
                logging.debug('bottomed out at replaced output: {0}'.format(
                    node_output))
                output, output_mask = self._replace_tensors[node_output]
                return output, output_mask
            # Next check if the current node has already been rebuilt.
            elif node in self._finished_nodes.keys():
                logging.debug('reached finished node: {0}'.format(node))
                return self._finished_nodes[node]
            # Next check if one of the graph_inputs has been reached.
            mask_map = TensorDict()
            for input, mask in zip(graph_inputs, graph_input_masks):
                mask_map[input] = mask

            try:
                output_mask = mask_map[node_output]
                logging.debug('bottomed out at a model input')
                return node_output, output_mask
            except KeyError:
                # Otherwise recursively call this method on the inbound nodes.
                inbound_nodes = parent_nodes(node)
                logging.debug('inbound_layers: {0}'.format(
                    [node.outbound_layer.name for node in inbound_nodes]))
                # Recursively rebuild the model up to `node`s inbound nodes to
                # obtain its inputs and input masks
                inputs, input_masks = zip(
                    *[_rebuild_rec(n) for n in inbound_nodes])

                if all(i is None for i in inputs):
                    output = None
                    try:
                        assert len(node.output_tensors) <= 1
                    except AssertionError as e:
                        raise e
                    except:
                        pass

                    output_mask = np.zeros(node.output_tensors.shape[1:], dtype=bool)
                elif any(i is None for i in inputs):
                    if node.outbound_layer.__class__.__name__ != 'Concatenate':
                        TypeError('Inputs can only be missing for concatenate layers.')
                    # remove Nones from inputs list
                    inputs = [i for i in inputs if i is not None]
                    new_layer, output_mask = self._apply_delete_mask(node, input_masks)
                    if len(inputs) == 1:
                        output = single_element(list(inputs))
                    else:
                        output = new_layer(single_element(list(inputs)))
                else:
                    new_layer, output_mask = self._apply_delete_mask(node, input_masks)
                    output = new_layer(single_element(list(inputs)))

                # Record that this node has been rebuild
                self._finished_nodes[node] = (output, output_mask)
                logging.debug('layer complete: {0}'.format(layer.name))
                return output, output_mask

        # Call the recursive _rebuild_rec method to rebuild the submodel up to
        # each output layer
        outputs, output_masks = zip(*[_rebuild_rec(n) for n in output_nodes])
        return single_element(outputs), output_masks

    def _delete_layer(self, node, inputs, input_masks):
        """Skip adding node.outbound_layer when building the graph."""
        # Skip the deleted layer by replacing its outputs with it inputs
        if not isinstance(inputs, tf.Tensor) and len(inputs) >= 2:
            raise ValueError('Cannot insert new layer at node with multiple '
                             'inbound layers.')
        inputs = single_element(inputs)
        input_masks = single_element(input_masks)
        deleted_layer_output = single_element(node.output_tensors)
        self._replace_tensors[deleted_layer_output] = (inputs, input_masks)

    def _insert_layer(self, node, inputs, input_masks, new_layer=None):
        """Insert new_layer into the graph before node.outbound_layer."""
        # This will not work for nodes with multiple inbound layers
        if not isinstance(inputs, tf.Tensor) and len(inputs) >= 2:
            raise ValueError('Cannot insert new layer at node with multiple '
                             'inbound layers.')
        # Call the new layer on the inbound layer's output
        new_output = new_layer(single_element(inputs))
        # Replace the inbound layer's output with the new layer's output
        old_output = get_one_tensor(node.input_tensors)
        input_masks = single_element(input_masks)
        self._replace_tensors[old_output] = (new_output, input_masks)

    def _replace_layer(self, node, inputs, input_masks, new_layer=None):
        """Replace node.outbound_layer with new_layer. Add it to the graph."""
        # Call the new layer on the rebuild submodel's inputs
        new_output = new_layer(single_element(inputs))

        # Replace the original layer's output with the new layer's output
        replaced_layer_output = single_element(node.output_tensors)
        input_masks = single_element(input_masks)
        self._replace_tensors[replaced_layer_output] = (new_output, input_masks)

    def _delete_channels(self, node, inputs, input_masks, channels=None, layer_name=None):
        """Delete selected channels of node.outbound_layer. Add it to the graph.
        """
        old_layer = node.outbound_layer
        old_layer_output = single_element(node.output_tensors)
        # Create a mask to propagate the deleted channels to downstream layers
        new_delete_mask = self._make_delete_mask(old_layer, channels)

        if len(set(channels)) == getattr(old_layer, get_channels_attr(old_layer)):
            self._replace_tensors[old_layer_output] = (None, new_delete_mask)
            return None

        # If this layer has already been operated on, use the cached copy of
        # the new layer. Otherwise, apply the inbound delete mask and
        # delete channels to obtain the new layer
        if old_layer in self._new_layers_map.keys():
            new_layer = self._new_layers_map[old_layer]
        else:
            temp_layer, new_mask = self._apply_delete_mask(node, input_masks)
            # This call is needed to initialise input_shape and output_shape
            temp_layer(single_element(inputs))
            new_layer = self._delete_channel_weights(temp_layer, channels)
            if layer_name:
                new_layer.name = layer_name
            self._new_layers_map[old_layer] = new_layer
        new_output = new_layer(single_element(inputs))
        # Replace the original layer's output with the modified layer's output
        self._replace_tensors[old_layer_output] = (new_output, new_delete_mask)

    def _apply_delete_mask(self, node, inbound_masks):
        """Apply the inbound delete mask and return the outbound delete mask
        When specific channels in a layer or layer instance are deleted, the
        mask propagates information about which channels are affected to
        downstream layers.
        If the layer contains weights, those which were previously connected
        to the deleted channels are deleted and outbound masks are set to None
        since further downstream layers aren't affected.
        If the layer does not contain weights, its output mask is calculated to
        reflect any transformations performed by the layer to ensure that
        information about the deleted channels is propagated downstream.
        Arguments:
            node(Node): The node where the delete mask is applied.
            inbound_masks: Mask(s) from inbound node(s).
        Returns:
            new_layer: Pass through `layer` if it has no weights, otherwise a
                       new `Layer` object with weights corresponding to the
                       inbound mask deleted.
            outbound_mask: Mask corresponding to `new_layer`.
        """

        # if delete_mask is None or all values are True, it does not affect
        # this layer or any layers above/downstream from it
        layer = node.outbound_layer
        if all(mask is None for mask in inbound_masks):
            new_layer = layer
            outbound_mask = None
            return new_layer, outbound_mask

        # If one or more of the masks are None, replace them with ones.
        if any(mask is None for mask in inbound_masks):
            inbound_masks = [np.ones(shape[1:], dtype=bool)
                             if inbound_masks[i] is None else inbound_masks[i]
                             for i, shape in enumerate(node.input_shapes)]

        # If the layer is shared and has already been affected by this
        # operation, use the cached new layer.
        if len(layer.inbound_nodes) > 1 \
                and layer in self._replace_layers_map.keys():
            return self._replace_layers_map[layer]

        output_shape = single_element(node.output_shapes)
        input_shape = single_element(node.input_shapes)
        data_format = getattr(layer, 'data_format', 'channels_last')
        inbound_masks = single_element(inbound_masks)
        # otherwise, delete_mask.shape should be: layer.input_shape[1:]
        layer_class = layer.__class__.__name__
        if layer_class == 'InputLayer':
            raise RuntimeError('This should never get here!')

        elif layer_class == 'Dense':
            if np.all(inbound_masks):
                new_layer = layer
            else:
                weights = layer.get_weights()
                weights[0] = weights[0][np.where(inbound_masks)[0], :]
                config = layer.get_config()
                config['weights'] = weights
                new_layer = type(layer).from_config(config)
            outbound_mask = None

        elif layer_class == 'Flatten':
            outbound_mask = np.reshape(inbound_masks, [-1, ])
            new_layer = layer

        elif layer_class in ('Conv1D', 'Conv2D', 'Conv3D'):
            if np.all(inbound_masks):
                new_layer = layer
            else:
                if data_format == 'channels_first':
                    inbound_masks = np.swapaxes(inbound_masks, 0, -1)
                # Conv layer: trim down inbound_masks to filter shape
                k_size = layer.kernel_size
                index = [slice(None, 1, None) for _ in k_size]
                inbound_masks = inbound_masks[tuple(index + [slice(None)])]
                weights = layer.get_weights()
                # Delete unused weights to obtain new_weights
                # Each deleted channel was connected to all of the channels
                # in layer; therefore, the mask must be repeated for each
                # channel.
                # `delete_mask`'s size: size(weights[0])
                delete_mask = np.tile(inbound_masks[..., np.newaxis], list(k_size) + [1, weights[0].shape[-1]])
                new_shape = list(weights[0].shape)
                new_shape[-2] = -1  # Weights always have channels_last
                weights[0] = np.reshape(weights[0][delete_mask], new_shape)
                # Instantiate new layer with new_weights
                config = layer.get_config()
                config['weights'] = weights
                new_layer = type(layer).from_config(config)
            outbound_mask = None

        elif layer_class in ('Cropping1D', 'Cropping2D', 'Cropping3D',
                             'MaxPooling1D', 'MaxPooling2D',
                             'MaxPooling3D',
                             'AveragePooling1D', 'AveragePooling2D',
                             'AveragePooling3D'):
            index = [slice(None, x, None) for x in output_shape[1:]]
            if data_format == 'channels_first':
                index[0] = slice(None)
            elif data_format == 'channels_last':
                index[-1] = slice(None)
            else:
                raise ValueError('Invalid data format')
            outbound_mask = inbound_masks[tuple(index)]
            new_layer = layer

        elif layer_class in ('UpSampling1D',
                             'UpSampling2D',
                             'UpSampling3D',
                             'ZeroPadding1D',
                             'ZeroPadding2D',
                             'ZeroPadding3D'):

            # Get slice of mask with all singleton dimensions except
            # channels dimension
            index = [slice(1)] * (len(input_shape) - 1)
            tile_shape = list(output_shape[1:])
            if data_format == 'channels_first':
                index[0] = slice(None)
                tile_shape[0] = 1
            elif data_format == 'channels_last':
                index[-1] = slice(None)
                tile_shape[-1] = 1
            else:
                raise ValueError('Invalid data format')
            channels_vector = inbound_masks[tuple(index)]
            # Tile this slice to create the outbound mask
            outbound_mask = np.tile(channels_vector, tile_shape)
            new_layer = layer

        elif layer_class in ('GlobalMaxPooling1D',
                             'GlobalMaxPooling2D',
                             'GlobalAveragePooling1D',
                             'GlobalAveragePooling2D'):
            # Get slice of mask with all singleton dimensions except
            # channels dimension
            index = [0] * (len(input_shape) - 1)
            if data_format == 'channels_first':
                index[0] = slice(None)
            elif data_format == 'channels_last':
                index[-1] = slice(None)
            else:
                raise ValueError('Invalid data format')
            channels_vector = inbound_masks[tuple(index)]
            # Tile this slice to create the outbound mask
            outbound_mask = channels_vector
            new_layer = layer

        elif layer_class in ('Dropout',
                             'Activation',
                             'SpatialDropout1D',
                             'SpatialDropout2D',
                             'SpatialDropout3D',
                             'ActivityRegularization',
                             'Masking',
                             'LeakyReLU',
                             'ELU',
                             'ThresholdedReLU',
                             'GaussianNoise',
                             'GaussianDropout',
                             'AlphaDropout',
                             'ReLU'):
            # Pass-through layers
            outbound_mask = inbound_masks
            new_layer = layer

        elif layer_class == 'Reshape':
            outbound_mask = np.reshape(inbound_masks, layer.target_shape)
            new_layer = layer

        elif layer_class == 'Permute':
            outbound_mask = np.transpose(inbound_masks,
                                         [x-1 for x in layer.dims])
            new_layer = layer

        elif layer_class == 'RepeatVector':
            outbound_mask = np.repeat(
                np.expand_dims(inbound_masks, 0),
                layer.n,
                axis=0)
            new_layer = layer

        elif layer_class == 'Embedding':
            # Embedding will always be the first layer so it doesn't need
            # to consider the inbound_delete_mask
            if inbound_masks is not None:
                raise ValueError('Channels cannot be deleted bedore Embedding '
                                 'layers because they change the number of '
                                 'channels.')
            outbound_mask = None
            new_layer = layer

        elif layer_class in ('Add', 'Multiply', 'Average', 'Maximum'):
            # The inputs must be the same size
            if not all_equal(inbound_masks):
                ValueError(
                    '{0} layers must have the same size inputs. All '
                    'inbound nodes must have the same channels deleted'
                    .format(layer_class))
            outbound_mask = inbound_masks[1]
            new_layer = layer

        elif layer_class == 'Concatenate':
            axis = layer.axis
            if layer.axis < 0:
                axis = axis % len(layer.input_shape[0])
            # Below: axis=axis-1 because the mask excludes the batch dimension
            outbound_mask = np.concatenate(inbound_masks, axis=axis-1)
            new_layer = layer

        elif layer_class in ('SimpleRNN', 'GRU', 'LSTM'):
            if np.all(inbound_masks):
                new_layer = layer
            else:
                weights = layer.get_weights()
                weights[0] = weights[0][np.where(inbound_masks[0, :])[0], :]
                config = layer.get_config()
                config['weights'] = weights
                new_layer = type(layer).from_config(config)
            outbound_mask = None

        elif layer_class == 'BatchNormalization':
            outbound_mask = inbound_masks
            # Get slice of mask with all singleton dimensions except
            # channels dimension
            index = [0] * (len(input_shape))
            assert len(layer.axis) == 1
            index[layer.axis[0]] = slice(None)
            index = index[1:]
            # TODO: Maybe use channel indices everywhere instead of masks?
            channel_indices = np.where(inbound_masks[tuple(index)] == False)[0]
            weights = [np.delete(w, channel_indices, axis=-1)
                       for w in layer.get_weights()]
            new_layer = BatchNormalization.from_config(
                layer.get_config())
            new_input_shape = list(input_shape)
            assert len(new_layer.axis) == 1
            new_input_shape[new_layer.axis[0]] -= len(channel_indices)
            print(new_input_shape)
            new_layer.build(new_input_shape)
            new_layer.set_weights(weights)

        elif layer_class == 'GroupNormalization':
            outbound_mask = inbound_masks
            # Get slice of mask with all singleton dimensions except
            # channels dimension
            index = [0] * (len(input_shape))
            axis = layer.axis
            if axis < 0:
                axis = axis % len(input_shape)
            index[axis] = slice(None)
            index = index[1:]
            # TODO: Maybe use channel indices everywhere instead of masks?
            channel_indices = np.where(inbound_masks[tuple(index)] == False)[0]
            weights = [np.delete(w, channel_indices, axis=-1)
                       for w in layer.get_weights()]
            new_layer = tfa.layers.GroupNormalization.from_config(
                layer.get_config())
            new_input_shape = list(input_shape)
            axis = new_layer.axis
            if axis < 0:
                axis = axis % len(input_shape)
            new_input_shape[axis] -= len(channel_indices)
            new_layer.build(new_input_shape)
            new_layer.set_weights(weights)

        else:
            # Not implemented:
            # - Lambda
            # - SeparableConv2D
            # - Conv2DTranspose
            # - LocallyConnected1D
            # - LocallyConnected2D
            # - TimeDistributed
            # - Bidirectional
            # - Dot
            # - PReLU
            # Warning/error needed for Reshape if channels axis is split
            raise ValueError('"{0}" layers are currently '
                             'unsupported.'.format(layer_class))

        if len(layer.inbound_nodes) > 1 and new_layer != layer:
            self._replace_layers_map[layer] = (new_layer, outbound_mask)

        return new_layer, outbound_mask

    def _delete_channel_weights(self, layer, channel_indices):
        """Delete channels from layer and remove the corresponding weights. utils
        Arguments:
            layer: A layer whose channels are to be deleted
            channel_indices: The indices of the channels to be deleted.
        Returns:
            A new layer with the channels and corresponding weights deleted.
        """
        layer_config = layer.get_config()
        channels_attr = get_channels_attr(layer)
        channel_count = layer_config[channels_attr]
        # Check inputs
        if any([i + 1 > channel_count for i in channel_indices]):
            raise ValueError('Channels_index value(s) out of range. '
                             'This layer only has {0} channels.'
                             .format(channel_count))
        print('Deleting {0}/{1} channels from layer: {2}'.format(
            len(channel_indices), channel_count, layer.name))
        # numpy.delete ignores negative indices in lists: wrap indices
        channel_indices = [i % channel_count for i in channel_indices]

        # Reduce layer channel count in config.
        layer_config[channels_attr] -= len(channel_indices)

        # Delete weights corresponding to deleted channels from config.
        # Except for recurrent layers, the weights' channels dimension is last.
        # Each recurrent layer type has a different internal weights layout.
        if layer.__class__.__name__ == 'SimpleRNN':
            weights = [np.delete(w, channel_indices, axis=-1)
                       for w in layer.get_weights()]
            weights[1] = np.delete(weights[1], channel_indices, axis=0)
        elif layer.__class__.__name__ == 'GRU':
            # Repeat the channel indices for all internal GRU weights.
            channel_indices_gru = [layer.units * m + i for m in range(3)
                                   for i in channel_indices]
            weights = [np.delete(w, channel_indices_gru, axis=-1)
                       for w in layer.get_weights()]
            weights[1] = np.delete(weights[1], channel_indices, axis=0)
        elif layer.__class__.__name__ == 'LSTM':
            # Repeat the channel indices for all interal LSTM weights.
            channel_indices_lstm = [layer.units * m + i for m in range(4)
                                    for i in channel_indices]
            weights = [np.delete(w, channel_indices_lstm, axis=-1)
                       for w in layer.get_weights()]
            weights[1] = np.delete(weights[1], channel_indices, axis=0)
        else:
            weights = [np.delete(w, channel_indices, axis=-1)
                       for w in layer.get_weights()]
        layer_config['weights'] = weights

        # Create new layer from the modified configuration and return it.
        return type(layer).from_config(layer_config)

    def _make_delete_mask(self, layer, channel_indices):
        """Make the boolean delete mask for layer's output deleting channels.
        The mask is used to remove the weights of the downstream layers which
        were connected to channels which have been deleted in this layer.
        The mask is a boolean array with the same size as the layer output
        excluding the first (batch) dimension.
        All elements of the mask corresponding to the removed channels are set
        to False. Other elements are set to True.
        Arguments:
            layer: A layer
            channel_indices: The indices of the channels to be deleted.
        Returns:
            A Numpy array of booleans of the same size as the output of layer
            excluding the batch dimension.
        """
        data_format = getattr(layer, 'data_format', 'channels_last')
        new_delete_mask = np.ones(layer.output_shape[1:], dtype=bool)
        if data_format == 'channels_first':
            new_delete_mask[channel_indices, ...] = False
        elif data_format == 'channels_last':
            new_delete_mask[..., channel_indices] = False
        else:
            ValueError('Invalid data_format property value')
        return new_delete_mask


def delete_channels(model, layer, channels, *, node_indices=None, copy=None):
    """Delete channels from instances of the specified layer.
    This method is designed to facilitate research into pruning networks to
    improve their prediction performance and/or reduce computational load by
    deleting channels.
    All weights associated with the deleted channels in the specified layer
    and any affected downstream layers are deleted.
    If the layer is shared and node_indices is set, channels will be deleted
    from the corresponding layer instances only. This will break the weight
    sharing between affected and unaffected instances in subsequent training.
    In this case affected instances will be renamed.
    Args:
        model: Model object.
        layer: Layer whose channels are to be deleted.
        channels: Indices of the channels to be deleted
        node_indices: Indices of the nodes where channels are to be deleted.
        copy: If True, the model will be copied before and after
              manipulation. This keeps both the old and new models' layers
              clean of each-others data-streams.
    Returns:
        A new Model with the specified channels and associated weights deleted.
    Notes:
        Channels are filters in conv layers and units in other layers.
    """
    surgeon = Surgeon(model, copy)
    surgeon.add_job('delete_channels', layer, node_indices=node_indices, channels=channels)
    return surgeon.operate()
