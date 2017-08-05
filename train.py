from keras.applications.xception import Xception, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

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

    # hyper parameters
    batch_size = 16
    num_classes = 102
    epochs = 200

    # Instantiate model
    model = Xception(include_top=True, weights=None, classes=num_classes)

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
    hist = model.fit(
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

    # save graphs
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    plt.plot(range(epochs), loss, marker='.', label='acc')
    plt.plot(range(epochs), val_loss, marker='.', label='val_acc')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.savefig('acc_xception.png')
    plt.clf()

    plt.plot(range(epochs), acc, marker='.', label='loss')
    plt.plot(range(epochs), val_acc, marker='.', label='val_loss')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('loss_xception.png')
    plt.clf()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
