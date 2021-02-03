"""The models subpackage contains definitions for the following model
architectures:
-  `ResNeXt` for CIFAR10 CIFAR100
You can construct a model with random weights by calling its constructor:
.. code:: python
    import models
    resnext29_16_64 = models.ResNeXt29_16_64(num_classes)
    resnext29_8_64 = models.ResNeXt29_8_64(num_classes)
    resnet20 = models.ResNet20(num_classes)
    resnet32 = models.ResNet32(num_classes)


.. ResNext: https://arxiv.org/abs/1611.05431
"""

from .imagenet_resnet import resnet50
