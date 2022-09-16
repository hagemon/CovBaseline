## Introduction

Implements of Neuron Coverage, K-multisection Coverage, Contribution Coverage and a alternative Multi-layer-section Coverage.

Experiments are deployed on MINIST **for now,** with a simple fully-connected neuron network:

| **Input** | **FC1** | **FC2** | **FC3** | **Output** |
| --------- | ------- | ------- | ------- | ---------- |
| 784       | 512     | 256     | 64      | 10         |

We discovered the status space size of each method, and made a comparison on MINIST.

### Status Space Size

| **Method (k=5)**             | **# Status** |
| ---------------------------- | ------------ |
| Neuron Coverage              | 842          |
| K-multisection Coverage      | 4210         |
| Contribution Coverage        | 148096       |
| Multi-layer Coverage         | 592384       |
| Multi-layer-section Coverage | 3702400      |

### Coverage

The comparison of coverage trends are shown below:

<img src="https://tva1.sinaimg.cn/large/e6c9d24ely1h68n5b511xj20m40gc0ty.jpg" alt="comarison" style="zoom:50%;" />

The comparison of the influence of k are shown below:

<img src="https://tva1.sinaimg.cn/large/e6c9d24ely1h68n7veai4j20my0gg75c.jpg" alt="k-influnce" style="zoom:50%;" />





