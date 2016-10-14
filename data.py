from keras.datasets import mnist
from keras.utils import np_utils


def load_data():

    rows, cols = 28, 28
    nb_classes = 10

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshape and format input
    X_train = X_train.reshape(X_train.shape[0], 1, rows, cols)
    X_test = X_test.reshape(X_test.shape[0], 1, rows, cols)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255.0
    X_test /= 255.0

    # Hot encoding
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    return (X_train, y_train, X_test, y_test)
