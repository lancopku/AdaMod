# Demos

Here we provide some demos of using AdaMod on several benchmark tasks.The purpose of these demos is to give an example of how to use it your research, and also illustrate the robust performance of AdaMod.

In short, AdaMod restrict the adaptive learning rates with adaptive and momental upper bounds. In this way, it can **smooths out unexpected large learning rates and stabilizes the training of deep neural networks.**.

In NMT examples, you can observe that AdaMod achieves both faster convergence and stronger performance compared with vanilla Adam when training Transfomer-based models even if **without warmup**. Other auxiliary examples prove the versatility of our method.