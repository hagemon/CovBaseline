import matplotlib.pyplot as plt


def coverage_line(coverages):
    x = [i+1 for i in range(len(coverages[0]))]
    for cov in coverages:
        plt.plot(x, cov)
    plt.show()
