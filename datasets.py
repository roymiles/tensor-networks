import tensorflow as tf

""" Loading datasets, data augmentation etc """


def load_cifar10():
    print('Loading cifar10 data ...')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    return (x_train/255., y_train), (x_test/255., y_test)


def load_cifar100():
    print('Loading cifar100 data ...')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)

    return (x_train/255., y_train), (x_test/255., y_test)