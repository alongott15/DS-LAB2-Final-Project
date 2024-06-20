"""
    Implementation of ResNet as described in the following article:
        https://arxiv.org/abs/1512.03385
    ResNet is a NN using a sequence of conv layers and residual blocks.
    Each conv layer contains multiple convolution kernels that are used for
    feature extraction.
    A residual block is a structure that contains 2 convolutional layers and 
    a residual connection.
    The first layer of the residual block is used for feature extraction while
    the second one is used to learn the residual.
    The residual connection is used to add the input to the output, to be able
    to facilitate the gradient propagation.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

class BasicBlock(tf.keras.Model):
    # 2 3x3 Conv Layers
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=3, strides=stride, padding='same', kernel_initializer='he_normal')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, padding='same', kernel_initializer='he_normal')
        self.bn2 = layers.BatchNormalization()
        self.downsample = downsample
        self.stride = stride

    def call(self, x, training=False):
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        if self.downsample is not None:
            shortcut = self.downsample(x, training=training)
        out += shortcut
        out = self.relu(out)
        return out

class Bottleneck(tf.keras.Model):
    # 3 Conv layers - 1x1, 3x3, 1x1
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=1, kernel_initializer='he_normal')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=stride, padding='same', kernel_initializer='he_normal')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(out_channels * self.expansion, kernel_size=1, kernel_initializer='he_normal')
        self.bn3 = layers.BatchNormalization()
        self.downsample = downsample
        self.stride = stride

    def call(self, x, training=False):
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, training=training)

        if self.downsample is not None:
            shortcut = self.downsample(x, training=training)
        out += shortcut
        out = self.relu(out)
        return out

class ResNet(tf.keras.Model):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = layers.Conv2D(self.inplanes, kernel_size=7, strides=2, padding='same', kernel_initializer='he_normal')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.maxpool = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')

        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes, kernel_initializer='he_normal')

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = models.Sequential([
                layers.Conv2D(planes * block.expansion, kernel_size=1, strides=stride, use_bias=False, kernel_initializer='he_normal'),
                layers.BatchNormalization()
            ])

        layers_list = []
        layers_list.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers_list.append(block(self.inplanes, planes))

        return models.Sequential(layers_list)

    def call(self, x, training=False):
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out, training=training)
        out = self.layer2(out, training=training)
        out = self.layer3(out, training=training)
        out = self.layer4(out, training=training)

        out = self.avgpool(out)
        out = self.fc(out)
        return out

def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

def resnet152(num_classes=1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)

if __name__ == "__main__":
    x = tf.random.normal([2, 224, 224, 3])
    model = resnet50()
    y = model(x)
    print(y.shape)
