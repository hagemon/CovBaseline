from train_op import train
from data_op import load_data
from model_op import get_model, save_model
from neuron_op import neuron_coverage, k_multi_section_coverage, contribution_coverage, multi_layer_coverage
from visual_op import coverage_line
from utils import tok


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()  # load data.
    model = get_model()  # get model if we have trained one, saved as 'model.h5'.
    if not model:
        # train and store a model.
        model = train(x_train, x_test, y_train, y_test)
        save_model(model)
    else:
        model.summary()
    tok()
    coverage1 = neuron_coverage(model, x_test)
    tok()
    coverage2 = k_multi_section_coverage(model, x_train, x_test)
    tok()
    coverage3 = contribution_coverage(model, x_test)
    tok()
    coverage4 = multi_layer_coverage(model, x_test)
    tok()
    coverage_line({
        'neuron coverage': coverage1,
        'k-multisection': coverage2,
        'contribution': coverage3,
        'multi-layer-coverage': coverage4
    })
