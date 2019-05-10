import tensorflow as tf
import numpy as np
from layers import *
from architectures import *
from tensor_network import TensorNetV1

conv_ranks = [5, 5, 5, 5]
fc_ranks = [3, 3, 3]

# NOTE: Just change architecture here
architecture = CIFAR100Example()
print("Number of parameters before = {}".format(architecture.num_parameters()))

model = TensorNetV1(architecture=architecture,
                    conv_ranks=conv_ranks,
                    fc_ranks=fc_ranks)

# Load the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

# The compile step specifies the training configuration.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs.
model.fit(x_train, y_train, batch_size=32, epochs=5)