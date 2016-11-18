from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D


def get_model():

    # Model parameters
    rows, cols = 28, 28
    input_shape = (1, rows, cols)

    nb_filters = 32
    nb_classes = 10

    kernel_size = (3, 3)
    pool_size = (2, 2)

    dense_layer_size = 128

    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='same',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(dense_layer_size))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    print model.summary()

    return model


if __name__ == '__main__':

    model = get_model()
