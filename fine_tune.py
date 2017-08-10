from keras.applications.xception import Xception
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import pickle as pkl

current_directory = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument(
    '--epochs_pre',
    type=int,
    default=10,
)
parser.add_argument(
    '--epochs_fine',
    type=int,
    default=200,
)
parser.add_argument(
    '--batch_size_pre',
    type=int,
    default=32,
)
parser.add_argument(
    '--batch_size_fine',
    type=int,
    default=16,
)
parser.add_argument(
    '--num_classes',
    type=int,
    default=102,
)
parser.add_argument(
    '--dataset_path',
    default=os.path.join(current_directory, 'dataset'),
)

def main(args):

    # hyper parameters
    batch_size_fine = args.batch_size_fine
    batch_size_pre = args.batch_size_pre
    num_classes = args.num_classes
    epochs_fine = args.epochs_fine
    epochs_pre = args.epochs_pre
    epochs = epochs_pre + epochs_fine
    dataset_path = args.dataset_path

    # create the pre-trained model
    base_model = Xception(include_top=False, weights='imagenet')

    # add a global average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.inputs, outputs=predictions)

    # prepare data
    x_train = np.load(os.path.join(dataset_path, 'x_train.npy'))
    y_train = np.load(os.path.join(dataset_path, 'y_train.npy'))
    x_test = np.load(os.path.join(dataset_path, 'x_test.npy'))
    y_test = np.load(os.path.join(dataset_path, 'y_test.npy'))

    # first: train only the top layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile model
    model.compile(
        loss=categorical_crossentropy,
        optimizer=SGD(lr=1e-2, momentum=0.9),
        metrics=['accuracy']
    )

    # train the top layers
    hist1 = model.fit(
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
        optimizer=SGD(lr=1e-3, momentum=0.9),
        loss=categorical_crossentropy,
        metrics=['accuracy'])

    # train the whole model
    hist2 = model.fit(
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

    # concatinate plot data
    acc = hist1.history['acc']
    val_acc = hist1.history['val_acc']
    loss = hist1.history['loss']
    val_loss = hist1.history['val_loss']

    acc.extend(hist2.history['acc'])
    val_acc.extend(hist2.history['val_acc'])
    loss.extend(hist2.history['loss'])
    val_loss.extend(hist2.history['val_loss'])

    # save graph image
    if os.path.exists(os.path.join(current_directory, 'result')) == False:
        os.mkdir(os.path.join(current_directory, 'result'))

    plt.plot(range(epochs), acc, marker='.', label='acc')
    plt.plot(range(epochs), val_acc, marker='.', label='val_acc')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.savefig(os.path.join(current_directory, 'result', 'acc.png'))
    plt.clf()

    plt.plot(range(epochs), loss, marker='.', label='loss')
    plt.plot(range(epochs), val_loss, marker='.', label='val_loss')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(current_directory, 'result', 'loss.png'))
    plt.clf()

    # save plot data as pickle file
    plot = {
        'acc': acc,
        'val_acc': val_acc,
        'loss': loss,
        'val_loss': val_loss,
    }
    with open(os.path.join(current_directory, 'result', 'plot_xception.dump'), 'wb') as f:
        pkl.dump(plot, f)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
