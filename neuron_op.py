import numpy as np
from tensorflow import keras


def get_output(model, data):
    extractor = keras.Model(inputs=model.inputs,
                            outputs=[layer.output for layer in model.layers if layer.name.startswith('dense')])
    outputs = extractor(data)
    # for f in outputs:
    #     print(f.shape)
    return outputs


def neuron_coverage(model, data, t=0.5):
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
        layer_ceil.append(layer_numpy.max(axis=0))
        layer_floor.append(layer_numpy.min(axis=0))

    return None
