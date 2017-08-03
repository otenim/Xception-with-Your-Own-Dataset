from keras.applications.xception import Xception, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy
import numpy as np
import os
import argparse

current_directory = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument(
    '--train_path',
    default=os.path.join(current_directory, 'src', 'train.txt')
)
parser.add_argument(
    '--test_path',
    default=os.path.join(current_directory, 'src', 'test.txt')
)

def main(args):

    # Instantiate model
    model = Xception(include_top=True, weights=None, classes=102)

    # hyper parameters
    batch_size = 32
    num_classes = 102
    epochs = 100

    # prepare data
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
    x_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')


    # summary of the model
    model.summary()

    # compile model
    model.compile(
        loss=categorical_crossentropy,
        optimizer=Adadelta(),
        metrics=['accuracy']
    )

    # learning section
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test)
    )

    # evaluation section
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
