from keras.applications.xception import Xception, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from keras.optimizers import Adadelta, SGD
from keras.losses import categorical_crossentropy
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
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
    batch_size_fine = 16
    batch_size_pre = 32
    num_classes = 102
    epochs_fine = 100
    epochs_pre = 5

    # create the pre-trained model
    base_model = Xception(include_top=False, weights='imagenet')

    # add a global average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.inputs, outputs=predictions)

    # prepare data
    x_train = np.load(os.path.join(current_directory, 'x_train.npy'))
    y_train = np.load(os.path.join(current_directory, 'y_train.npy'))
    x_test = np.load(os.path.join(current_directory, 'x_test.npy'))
    y_test = np.load(os.path.join(current_directory, 'y_test.npy'))

    # first: train only the top layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile model
    model.compile(
        loss=categorical_crossentropy,
        optimizer=Adadelta(),
        metrics=['accuracy']
    )

    # train the top layers
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size_pre,
        epochs=epochs_pre,
        verbose=1,
        validation_data=(x_test, y_test)
    )

    # second: train all layers with lower learning rate
    for layer in model.layers:
        layer.trainable = True

    # recompile
    model.compile(
        optimizer=SGD(lr=1e-4, momentum=0.9),
        loss=categorical_crossentropy,
        metrics=['accuracy'])

    # train the whole model
    hist = model.fit(
        x_train,
        y_train,
        batch_size=batch_size_fine,
        epochs=epochs_fine,
        verbose=1,
        validation_data=(x_test, y_test)
    )

    # evaluation
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # save graphs
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    plt.plot(range(epochs), acc, marker='.', label='acc')
    plt.plot(range(epochs), val_acc, marker='.', label='val_acc')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.savefig(os.path.join(current_directory, 'acc_xception_fine.png'))
    plt.clf()

    plt.plot(range(epochs), loss, marker='.', label='loss')
    plt.plot(range(epochs), val_loss, marker='.', label='val_loss')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(current_directory, 'loss_xception_fine.png'))
    plt.clf()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
