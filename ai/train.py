import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
import keras

if __name__ == '__main__':
    batch_size = 128
    epochs = 100

    # Load the entire dataset
    with np.load("datasets/states.npz") as f:
        X = f['arr_0']
        Y = f['arr_1']

    np.random.seed(113)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    # split data
    cut = int(len(X) / 3)
    x_train = X[:cut]
    y_train = Y[:cut]
    x_validation = X[cut + 1:cut + cut]
    y_validation = Y[cut + 1:cut + cut]
    x_test = X[cut + cut + 1:]
    y_test = Y[cut + cut + 1:]

    y_train = keras.utils.to_categorical(y_train, 2)
    y_test = keras.utils.to_categorical(y_test, 2)
    y_validation = keras.utils.to_categorical(y_validation, 2)
    print("Total records: ", len(X))
    # Load the model
    model = Sequential()
    model.add(Conv2D(16, kernel_size=1, activation='relu', input_shape=X[0].shape))
    model.add(Conv2D(16, kernel_size=1, activation='relu'))
    model.add(Conv2D(32, kernel_size=1, activation='relu'))
    model.add(Conv2D(32, kernel_size=1, activation='relu'))
    model.add(Conv2D(64, kernel_size=1, activation='relu'))
    model.add(Conv2D(64, kernel_size=1, activation='relu'))
    model.add(Conv2D(128, kernel_size=1, activation='relu'))
    model.add(Conv2D(128, kernel_size=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_validation, y_validation))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save("datasets/model.h5")
