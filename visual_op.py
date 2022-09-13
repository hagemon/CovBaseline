import matplotlib.pyplot as plt


def coverage_line(coverages):
    for name in coverages:
        cov = coverages[name]
        x = [i+1 for i in range(len(cov))]
        plt.plot(x, cov, label=name)
        plt.legend()
    plt.show()
