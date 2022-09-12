from train_op import train
from data_op import load_data
from model_op import get_model, save_model
from neuron_op import neuron_coverage, k_multi_section_coverage
from visual_op import coverage_line


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()  # load data.
    model = get_model()  # get model if we have trained one, saved as 'model.h5'.
    if not model:
        # train and store a model.
        model = train(x_train, x_test, y_train, y_test)
        save_model(model)
    else:
        model.summary()
    coverage1 = neuron_coverage(model, x_test)
    coverage2 = k_multi_section_coverage(model, x_train, x_test)
    coverage_line([coverage1, coverage2])
