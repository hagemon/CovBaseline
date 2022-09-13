import numpy as np
from tensorflow import keras
from utils import tok

def get_output(model, data):
    extractor = keras.Model(inputs=model.inputs,
                            outputs=[layer.output for layer in model.layers if layer.name.startswith('dense')])
    outputs = extractor(data)
    # for f in outputs:
    #     print(f.shape)
    return outputs


def get_weights(model):
    return [layer.get_weights()[0] for layer in model.layers if layer.name.startswith('dense')]


def neuron_coverage(model, data, t=0.25):
    outputs = get_output(model, data)
    total_count = 0
    n_samples = outputs[0].shape[0]
    cum_count = np.array([0 for _ in range(n_samples)])
    for layer in outputs:
        act = (layer.numpy() > t).cumsum(axis=0) > 0
        act_count = act.sum(axis=1)
        cum_count += act_count
        total_count += layer.shape[1]
    coverage = cum_count / total_count
    return coverage


def k_multi_section_coverage(model, train, test, k=5):
    outputs = get_output(model, train)
    layer_ceil = []
    layer_floor = []

    for layer in outputs:
        layer_numpy = layer.numpy()
        max_val, min_val = layer_numpy.max(axis=0), layer_numpy.min(axis=0)
        const_ind = (max_val-min_val) == 0  # avoid divide by 0 exception
        max_val[const_ind] = 1
        min_val[const_ind] = 0
        layer_ceil.append(layer_numpy.max(axis=0))
        layer_floor.append(layer_numpy.min(axis=0))

    outputs = get_output(model, test)
    bits = np.array([2**i for i in range(k)], dtype='uint8')
    bits_range = np.array([i for i in range(2**k)], dtype='uint8')
    bits_ones = np.unpackbits(bits_range).reshape(2**k, 8).sum(-1)
    n_samples = outputs[0].shape[0]
    cum_count = np.array([0. for _ in range(n_samples)])
    total_count = 0
    for i, layer in enumerate(outputs):
        layer_numpy = layer.numpy()
        # n_neurons = layer_numpy.shape[1]
        max_val, min_val = layer_ceil[i], layer_floor[i]
        bit_ind = ((layer_numpy-min_val)/(max_val-min_val)*(k-1)).astype('int')
        # print(bit_ind)
        corners = (bit_ind < 0) | (bit_ind >= k)
        bit_ind[corners] = 0  # avoid indexing error
        bases = bits[bit_ind]
        bases[corners] = 0
        counter = np.bitwise_or.accumulate(bases)
        act_count = bits_ones[counter].sum(-1)
        cum_count += act_count
        total_count += layer_numpy.shape[1]*k

    coverage = cum_count / total_count
    return coverage


def contribution_coverage(model, data, t=0.6):
    outputs = get_output(model, data)
    weights = get_weights(model)
    layers = [layer.numpy() for layer in outputs]
    n_layers = len(layers)
    total_count = 0
    n_samples = outputs[0].shape[0]
    cum_count = np.array([0 for _ in range(n_samples)])
    for i in range(n_layers-1, 0, -1):
        pre, cur = layers[i-1], layers[i]
        weight = weights[i]
        if i == n_layers-1:
            layer_act = cur == cur.max(1, keepdims=True)
        else:
            layer_act = cur > t
        layer_inputs = (np.expand_dims(weight, axis=0)*np.expand_dims(pre, axis=2))
        inputs_max, inputs_min = layer_inputs.max(axis=1), layer_inputs.min(axis=1)
        normed = (layer_inputs - np.expand_dims(inputs_min, 1)) / np.expand_dims(inputs_max - inputs_min, 1)
        act_mask = ((normed * np.expand_dims(layer_act, axis=1)) > t)
        cum_count += np.bitwise_or.accumulate(act_mask).sum((-2, -1))
        total_count += weight.shape[0]*weight.shape[1]

    coverage = cum_count / total_count
    return coverage


def multi_layer_coverage(model, data, t=0.6):
    outputs = get_output(model, data)
    layers = [layer.numpy() for layer in outputs]
    n_layers = len(layers)
    bits = np.array([2 ** i for i in range(4)], dtype='uint8')
    bits_range = np.array([i for i in range(2 ** 4)], dtype='uint8')
    bits_ones = np.unpackbits(bits_range).reshape(2 ** 4, 8).sum(-1)
    total_count = 0.
    n_samples = outputs[0].shape[0]
    cum_count = np.array([0. for _ in range(n_samples)])
    pre = layers[0]
    pre_act = pre > t
    for i in range(1, n_layers):
        cur = layers[i]
        act = cur > t
        pre_ind = pre_act * 2
        cur_ind = act * 1
        status_mask = np.expand_dims(pre_ind, 2) + np.expand_dims(cur_ind, 1)
        counter = np.bitwise_or.accumulate(bits[status_mask])
        act_count = bits_ones[counter].sum((-2, -1))
        cum_count += act_count

        total_count += pre.shape[1]*cur.shape[1]*4
        pre, pre_act = cur, act

    coverage = cum_count / total_count
    return coverage
