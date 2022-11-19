import os
import datetime

import numpy as np
import tensorflow as tf

from utils import load_data, plot_history_tf, plot_heat_map

# project root path
project_path = "./"
# define log directory
# must be a subdirectory of the directory specified when starting the web application
# it is recommended to use the date time as the subdirectory name
log_dir = project_path + "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = project_path + "ecg_model.h5"

# the ratio of the test set
RATIO = 0.3
# the random seed
RANDOM_SEED = 42
BATCH_SIZE = 128
NUM_EPOCHS = 30


# build the CNN model
def build_model():
    newModel = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(300,)),
        # reshape the tensor with shape (batch_size, 300) to (batch_size, 300, 1)
        tf.keras.layers.Reshape(target_shape=(300, 1)),
        # the first convolution layer, 4 21x1 convolution kernels, output shape (batch_size, 300, 4)
        tf.keras.layers.Conv1D(filters=4, kernel_size=21, strides=1, padding='SAME', activation='relu'),
        # the first pooling layer, max pooling, pooling size=3 , stride=2, output shape (batch_size, 150, 4)
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # the second convolution layer, 16 23x1 convolution kernels, output shape (batch_size, 150, 16)
        tf.keras.layers.Conv1D(filters=16, kernel_size=23, strides=1, padding='SAME', activation='relu'),
        # the second pooling layer, max pooling, pooling size=3, stride=2, output shape (batch_size, 75, 16)
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # the third convolution layer, 32 25x1 convolution kernels, output shape (batch_size, 75, 32)
        tf.keras.layers.Conv1D(filters=32, kernel_size=25, strides=1, padding='SAME', activation='relu'),
        # the third pooling layer, average pooling, pooling size=3, stride=2, output shape (batch_size, 38, 32)
        tf.keras.layers.AvgPool1D(pool_size=3, strides=2, padding='SAME'),
        # the fourth convolution layer, 64 27x1 convolution kernels, output shape (batch_size, 38, 64)
        tf.keras.layers.Conv1D(filters=64, kernel_size=27, strides=1, padding='SAME', activation='relu'),
        # flatten layer, for the next fully connected layer, output shape (batch_size, 38*64)
        tf.keras.layers.Flatten(),
        # fully connected layer, 128 nodes, output shape (batch_size, 128)
        tf.keras.layers.Dense(128, activation='relu'),
        # Dropout layer, dropout rate = 0.2
        tf.keras.layers.Dropout(rate=0.2),
        # fully connected layer, 5 nodes (number of classes), output shape (batch_size, 5)
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    return newModel


def main():
    # X_train,y_train is the training set
    # X_test,y_test is the test set
    X_train, X_test, y_train, y_test = load_data(RATIO, RANDOM_SEED)

    if os.path.exists(model_path):
        # import the pre-trained model if it exists
        print('Import the pre-trained model, skip the training process')
        model = tf.keras.models.load_model(filepath=model_path)
    else:
        # build the CNN model
        model = build_model()
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        # define the TensorBoard callback object
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # train and evaluate model
        history = model.fit(X_train, y_train, epochs=NUM_EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_data=(X_test, y_test),
                            callbacks=[tensorboard_callback])
        # save the model
        model.save(filepath=model_path)
        # plot the training history
        plot_history_tf(history)

    # predict the class of test data
    # y_pred = model.predict_classes(X_test)  # predict_classes has been deprecated
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    # plot confusion matrix heat map
    plot_heat_map(y_test, y_pred)


if __name__ == '__main__':
    main()
