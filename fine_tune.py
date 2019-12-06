import math
import os
import argparse
import matplotlib
import imghdr
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.xception import Xception, preprocess_input
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.losses import categorical_crossentropy
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

matplotlib.use('Agg')
current_directory = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('dataset_root')
parser.add_argument('classes')
parser.add_argument('result_root')
parser.add_argument('--epochs_pre', type=int, default=5)
parser.add_argument('--epochs_fine', type=int, default=50)
parser.add_argument('--batch_size_pre', type=int, default=32)
parser.add_argument('--batch_size_fine', type=int, default=16)
parser.add_argument('--lr_pre', type=float, default=1e-3)
parser.add_argument('--lr_fine', type=float, default=1e-4)
parser.add_argument('--snapshot_period_pre', type=int, default=1)
parser.add_argument('--snapshot_period_fine', type=int, default=1)
parser.add_argument('--split', type=float, default=0.8)


def generate_from_paths_and_labels(
        input_paths, labels, batch_size, input_size=(299, 299)):
    num_samples = len(input_paths)
    while 1:
        perm = np.random.permutation(num_samples)
        input_paths = input_paths[perm]
        labels = labels[perm]
        for i in range(0, num_samples, batch_size):
            inputs = list(map(
                lambda x: image.load_img(x, target_size=input_size),
                input_paths[i:i+batch_size]
            ))
            inputs = np.array(list(map(
                lambda x: image.img_to_array(x),
                inputs
            )))
            inputs = preprocess_input(inputs)
            yield (inputs, labels[i:i+batch_size])


def main(args):

    # ====================================================
    # Preparation
    # ====================================================
    # parameters
    epochs = args.epochs_pre + args.epochs_fine
    args.dataset_root = os.path.expanduser(args.dataset_root)
    args.result_root = os.path.expanduser(args.result_root)
    args.classes = os.path.expanduser(args.classes)

    # load class names
    with open(args.classes, 'r') as f:
        classes = f.readlines()
        classes = list(map(lambda x: x.strip(), classes))
    num_classes = len(classes)

    # make input_paths and labels
    input_paths, labels = [], []
    for class_name in os.listdir(args.dataset_root):
        class_root = os.path.join(args.dataset_root, class_name)
        class_id = classes.index(class_name)
        for path in os.listdir(class_root):
            path = os.path.join(class_root, path)
            if imghdr.what(path) is None:
                # this is not an image file
                continue
            input_paths.append(path)
            labels.append(class_id)

    # convert to one-hot-vector format
    labels = to_categorical(labels, num_classes=num_classes)

    # convert to numpy array
    input_paths = np.array(input_paths)

    # shuffle dataset
    perm = np.random.permutation(len(input_paths))
    labels = labels[perm]
    input_paths = input_paths[perm]

    # split dataset for training and validation
    border = int(len(input_paths) * args.split)
    train_labels = labels[:border]
    val_labels = labels[border:]
    train_input_paths = input_paths[:border]
    val_input_paths = input_paths[border:]
    print("Training on %d images and labels" % (len(train_input_paths)))
    print("Validation on %d images and labels" % (len(val_input_paths)))

    # create a directory where results will be saved (if necessary)
    if os.path.exists(args.result_root) is False:
        os.makedirs(args.result_root)

    # ====================================================
    # Build a custom Xception
    # ====================================================
    # instantiate pre-trained Xception model
    # the default input shape is (299, 299, 3)
    # NOTE: the top classifier is not included
    base_model = Xception(
        include_top=False,
        weights='imagenet',
        input_shape=(299, 299, 3))

    # create a custom top classifier
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.inputs, outputs=predictions)

    # ====================================================
    # Train only the top classifier
    # ====================================================
    # freeze the body layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile model
    model.compile(
        loss=categorical_crossentropy,
        optimizer=Adam(lr=args.lr_pre),
        metrics=['accuracy']
    )

    # train
    hist_pre = model.fit_generator(
        generator=generate_from_paths_and_labels(
            input_paths=train_input_paths,
            labels=train_labels,
            batch_size=args.batch_size_pre
        ),
        steps_per_epoch=math.ceil(
            len(train_input_paths) / args.batch_size_pre),
        epochs=args.epochs_pre,
        validation_data=generate_from_paths_and_labels(
            input_paths=val_input_paths,
            labels=val_labels,
            batch_size=args.batch_size_pre
        ),
        validation_steps=math.ceil(
            len(val_input_paths) / args.batch_size_pre),
        verbose=1,
        callbacks=[
            ModelCheckpoint(
                filepath=os.path.join(
                    args.result_root,
                    'model_pre_ep{epoch}_valloss{val_loss:.3f}.h5'),
                period=args.snapshot_period_pre,
            ),
        ],
    )
    model.save(os.path.join(args.result_root, 'model_pre_final.h5'))

    # ====================================================
    # Train the whole model
    # ====================================================
    # set all the layers to be trainable
    for layer in model.layers:
        layer.trainable = True

    # recompile
    model.compile(
        optimizer=Adam(lr=args.lr_fine),
        loss=categorical_crossentropy,
        metrics=['accuracy'])

    # train
    hist_fine = model.fit_generator(
        generator=generate_from_paths_and_labels(
            input_paths=train_input_paths,
            labels=train_labels,
            batch_size=args.batch_size_fine
        ),
        steps_per_epoch=math.ceil(
            len(train_input_paths) / args.batch_size_fine),
        epochs=args.epochs_fine,
        validation_data=generate_from_paths_and_labels(
            input_paths=val_input_paths,
            labels=val_labels,
            batch_size=args.batch_size_fine
        ),
        validation_steps=math.ceil(
            len(val_input_paths) / args.batch_size_fine),
        verbose=1,
        callbacks=[
            ModelCheckpoint(
                filepath=os.path.join(
                    args.result_root,
                    'model_fine_ep{epoch}_valloss{val_loss:.3f}.h5'),
                period=args.snapshot_period_fine,
            ),
        ],
    )
    model.save(os.path.join(args.result_root, 'model_fine_final.h5'))

    # ====================================================
    # Create & save result graphs
    # ====================================================
    # concatinate plot data
    acc = hist_pre.history['acc']
    val_acc = hist_pre.history['val_acc']
    loss = hist_pre.history['loss']
    val_loss = hist_pre.history['val_loss']
    acc.extend(hist_fine.history['acc'])
    val_acc.extend(hist_fine.history['val_acc'])
    loss.extend(hist_fine.history['loss'])
    val_loss.extend(hist_fine.history['val_loss'])

    # save graph image
    plt.plot(range(epochs), acc, marker='.', label='acc')
    plt.plot(range(epochs), val_acc, marker='.', label='val_acc')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.savefig(os.path.join(args.result_root, 'acc.png'))
    plt.clf()

    plt.plot(range(epochs), loss, marker='.', label='loss')
    plt.plot(range(epochs), val_loss, marker='.', label='val_loss')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(args.result_root, 'loss.png'))
    plt.clf()

    # save plot data as pickle file
    plot = {
        'acc': acc,
        'val_acc': val_acc,
        'loss': loss,
        'val_loss': val_loss,
    }
    with open(os.path.join(args.result_root, 'plot.dump'), 'wb') as f:
        pkl.dump(plot, f)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
