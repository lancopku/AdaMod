# Examples on CIFAR-100

In this example, we test AdaMod on the standard CIFAR-100 image classification dataset, comparing with SGD and Adam. The implementation is highly based on [this project](https://github.com/kuangliu/pytorch-cifar).

Tested with PyTorch 1.1.0.

## Settings

We have already provided the results produced by AdaMod with default settings and baseline optimizers with their best hyperparameters. The best hyperparameters are listed as follows to ease your reproduction:

**ResNet-34/DenseNet-121:**

| optimizer | lr | momentum | beta1 | beta2 | beta3 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| SGD | 0.05 | 0.9 | | | | |
| Adam | 0.001 | | 0.9 | 0.999 | | |
| AdaMod (def.) | 0.001 | | 0.9 | 0.999 | 0.999 |

For the sake of better performance, we apply a weight decay of `5e-4` to all the optimizers (decoupled weight decay to adaptive methods).

## Running by Yourself

You may also run the experiment and visualize the result by yourself. The following is an example to train DenseNet-121 using AdaMod with a learning rate of 0.001 and a smoothing coefficient (i.e. **beta3**) of 0.999.

```bash
python main.py --model=densenet --optim=adamod --lr=0.001 --beta3=0.999
```

The checkpoints will be saved in the `checkpoint` folder and the data points of the learning curve will be save in the `curve` folder.

## Visualization

You can directly run [visualization.py](./visualization.py) to make it easier to visualize the performance of AdaMod.

## Acknowledgement
The way of searching the best settings for baseline optimizers is referenced from Luo et al. (2019). [Adaptive Gradient Methods with Dynamic Bound of Learning Rate](https://openreview.net/forum?id=Bkg3g2R9FX). In *Proc. of ICLR 2019*.




