from neuron_op import *
from utils import tok
from visual_op import coverage_line
from model_op import get_model
from data_op import load_data

if __name__ == '__main__':
    model = get_model()
    x_train, x_test, y_train, y_test = load_data()
    # covs = []
    # ks = [2, 3, 4, 5]
    # for k in ks:
    #     tok()
    #     coverage = multi_layer_section_coverage(model, x_train, x_test, k=k)
    #     covs.append(coverage)
    # tok()
    # coverage_line(coverages={'MLSC k={}'.format(ks[i]): covs[i] for i in range(len(ks))})
    tok()
    c1 = k_multi_section_coverage(model, x_train, x_test, k=5)
    tok()
    c2 = k_multi_section_coverage_loop(model, x_train, x_test, k=5)
    tok()
    coverage_line({
        'k1': c1,
        'k2': c2
    })
