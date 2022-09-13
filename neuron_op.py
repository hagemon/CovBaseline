import numpy as np
from tensorflow import keras


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

    print('NC total:', total_count)
    coverage = cum_count / total_count
    return coverage


def k_multi_section_coverage(model, train, test, k=5):
    outputs = get_output(model, train)
    layer_max = []
    layer_min = []

    for layer in outputs:
        layer_numpy = layer.numpy()
        max_val, min_val = layer_numpy.max(axis=0), layer_numpy.min(axis=0)
        const_ind = (max_val-min_val) == 0  # avoid divide by 0 exception
        max_val[const_ind] = 1
        min_val[const_ind] = 0
        layer_max.append(layer_numpy.max(axis=0))
        layer_min.append(layer_numpy.min(axis=0))

    outputs = get_output(model, test)
    bits = np.array([2**i for i in range(k)], dtype='uint16')
    bits_range = np.array([i for i in range(2**k)], dtype='uint16')
    bits_ones = np.array([bin(d).count('1') for d in bits_range])
    n_samples = outputs[0].shape[0]
    cum_count = np.array([0. for _ in range(n_samples)])
    total_count = 0
    for i, layer in enumerate(outputs):
        layer_numpy = layer.numpy()
        max_val, min_val = layer_max[i], layer_min[i]
        bit_ind = ((layer_numpy-min_val)/(max_val-min_val)*k).astype('int')
        corners = (bit_ind < 0) | (bit_ind >= k)
        bit_ind[corners] = 0  # avoid indexing error
        bases = bits[bit_ind]
        bases[corners] = 0
        counter = np.bitwise_or.accumulate(bases)
        act_count = bits_ones[counter].sum(-1)
        cum_count += act_count
        total_count += layer_numpy.shape[1]*k

    print('kMSC total:', total_count)
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
        counter = np.zeros([weight.shape[0], weight.shape[1]], dtype='uint8')
        for s in range(n_samples):
            pre_sample = pre[s]
            layer_inputs = weight*np.expand_dims(pre_sample, axis=1)
            inputs_max, inputs_min = layer_inputs.max(0), layer_inputs.min(0)
            normed = (layer_inputs-inputs_min)/(inputs_max-inputs_min)
            act_mask = (normed*layer_act[s]) > t
            counter = counter | act_mask
            cum_count[s] += counter.sum()
        total_count += weight.shape[0]*weight.shape[1]

    print('CC total:', total_count)
    coverage = cum_count / total_count
    return coverage


def multi_layer_coverage(model, data, t=0.5):
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
        n1, n2 = pre.shape[1], cur.shape[1]
        cur_act = cur > t
        counter = np.zeros([n1, n2], dtype='uint8')
        for s in range(n_samples):
            pre_sample, cur_sample = pre_act[s]*2, cur_act[s]*1
            act_ind = np.expand_dims(pre_sample, 1) + np.expand_dims(cur_sample, 0)
            counter = counter | bits[act_ind]
            cum_count[s] += bits_ones[counter].sum()

        total_count += n1 * n2 * 4
        pre = cur
        pre_act = cur_act

    print('MLC total:', total_count)
    coverage = cum_count / total_count
    return coverage


def _window_coverage(model, data, t=0.5, window_size=3):
    """
    This algorithm is runs too slow, which is not feasible for practice.
    """
    outputs = get_output(model, data)
    layers = [layer.numpy() for layer in outputs]
    n_layers = len(layers)
    bits = np.array([2 ** i for i in range(2**window_size)], dtype='uint8')
    bits_range = np.array([i for i in range(bits[-1]*2)], dtype='uint8')
    bits_ones = np.unpackbits(bits_range).reshape(len(bits_range), 8).sum(-1)

    total_count = 0.
    n_samples = outputs[0].shape[0]
    cum_count = np.array([0. for _ in range(n_samples)])
    window_act = [layer > t for layer in layers[:window_size]]
    layer_neurons = [layer.shape[1] for layer in layers[:window_size]]
    for i in range(window_size, n_layers):
        counter = np.zeros(layer_neurons, dtype='uint8').flatten()
        for s in range(n_samples):
            window_samples = [samples[s]*(2**(window_size-i-1)) for i, samples in enumerate(window_act)]
            mask = window_samples[0]
            for samples in window_samples[1:]:
                mask = (np.expand_dims(mask, -1) + samples).flatten()
            counter = counter | bits[mask]
            cum_count[s] += bits_ones[counter].sum()

        total_count += np.multiply.reduce(layer_neurons)*(2**window_size)
        window_act = window_act[1:] + [layers[i] > 5]

    print('window total:', total_count)
    coverage = cum_count / total_count
    return coverage


def multi_layer_section_coverage(model, train, test, k=5):

    def get_neuron_norm(lay, l_max, l_min):
        normed = ((lay - l_min) / (l_max - l_min) * k).astype(int)
        c = (normed < 0) | (normed >= k)
        normed[c] = 0
        return normed

    outputs = get_output(model, train)
    # train set
    layer_max = []
    layer_min = []

    for layer in outputs:
        layer_numpy = layer.numpy()
        max_val, min_val = layer_numpy.max(axis=0), layer_numpy.min(axis=0)
        const_ind = (max_val-min_val) == 0  # avoid divide by 0 exception
        max_val[const_ind] = 1
        min_val[const_ind] = 0
        layer_max.append(layer_numpy.max(axis=0))
        layer_min.append(layer_numpy.min(axis=0))

    outputs = get_output(model, test)
    layers = [layer.numpy() for layer in outputs]
    n_layers = len(layers)
    bases = np.array([1 << i for i in range(k*k)], dtype='uint64')  # 2^0 to 2^10
    bits_range = np.array([i for i in range(int(bases[-1]*2))], dtype='uint64')  # from 0 to 2^11-1
    bits_ones = np.array([bin(d).count('1') for d in bits_range])

    total_count = 0.
    n_samples = outputs[0].shape[0]
    cum_count = np.array([0. for _ in range(n_samples)])
    pre_normed = get_neuron_norm(layers[0], layer_max[0], layer_min[0])
    for i in range(1, n_layers):
        cur_normed = get_neuron_norm(layers[i], layer_max[i], layer_min[i])
        n1, n2 = pre_normed.shape[1], cur_normed.shape[1]
        counter = np.zeros([n1, n2], dtype='uint64')

        for s in range(n_samples):
            pre_bin, cur_bin = pre_normed[s], cur_normed[s]
            act_ind = np.expand_dims(pre_bin*k, 1) + np.expand_dims(cur_bin, 0)
            counter = counter | bases[act_ind]
            cum_count[s] += bits_ones[counter].sum()

        total_count += n1 * n2 * (k**2)
        pre_normed = cur_normed

    print('MLSC total:', total_count)
    coverage = cum_count / total_count
    return coverage
