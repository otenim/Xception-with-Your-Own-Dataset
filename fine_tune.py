from keras.applications.xception import Xception, preprocess_input
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.losses import categorical_crossentropy
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import math
import numpy as np
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle as pkl
import datetime

current_directory = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument(
    'dataset_root',
)
parser.add_argument(
    'classes',
)
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
    '--split',
    type=float,
    default=0.8,
)
parser.add_argument(
    '--result_root',
    default=os.path.join(current_directory, 'result')
)


def generate_from_paths_and_labels(input_paths, labels, batch_size, input_size=(299,299)):

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

    # parameters
    batch_size_fine = args.batch_size_fine
    batch_size_pre = args.batch_size_pre
    epochs_fine = args.epochs_fine
    epochs_pre = args.epochs_pre
    epochs = epochs_pre + epochs_fine
    split = args.split
    dataset_root = os.path.abspath(args.dataset_root)
    result_root = os.path.abspath(args.result_root)
    classes_path = os.path.abspath(args.classes)
    with open(classes_path, 'r') as f:
        classes = f.readlines()
        classes = list(map(lambda x: x.strip(), classes))
    num_classes = len(classes)

    # make input_paths and labels
    input_paths, labels = [], []
    for class_name in os.listdir(dataset_root):
        class_root = os.path.join(dataset_root, class_name)
        class_id = classes.index(class_name)
        for img in os.listdir(class_root):
            if img.split('.')[-1] not in ['jpg', 'png']:
                continue
            path = os.path.join(class_root, img)
            input_paths.append(path)
            labels.append(class_id)
    labels = np.eye(num_classes, dtype=np.float32)[labels] # one-hot vectors
    input_paths = np.array(input_paths)

    # shuffle dataset
    perm = np.random.permutation(len(input_paths))
    labels = labels[perm]
    input_paths = input_paths[perm]

    # split dataset for training and validation
    border = int(len(input_paths) * split)
    train_labels, val_labels = labels[:border], labels[border:]
    train_input_paths, val_input_paths = input_paths[:border], input_paths[border:]
    print("Training on %d images and labels" % (len(train_input_paths)))
    print("Validation on %d images and labels" % (len(val_input_paths)))


    # create the pre-trained model
    base_model = Xception(include_top=False, weights='imagenet', input_shape=(299,299,3))

    # add a global average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.inputs, outputs=predictions)

    # first: train only the top layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile model
    model.compile(
        loss=categorical_crossentropy,
        optimizer=Adam(),
        metrics=['accuracy']
    )

    # train the top layers
    hist1 = model.fit_generator(
        generator=generate_from_paths_and_labels(
            input_paths=train_input_paths,
            labels=train_labels,
            batch_size=batch_size_pre
        ),
        steps_per_epoch=math.ceil(len(train_input_paths) / batch_size_pre),
        epochs=epochs_pre,
        validation_data=generate_from_paths_and_labels(
            input_paths=val_input_paths,
            labels=val_labels,
            batch_size=batch_size_pre
        ),
        validation_steps=math.ceil(len(val_input_paths) / batch_size_pre),
        verbose=1,
    )


    # second: train all layers with lower learning rate
    for layer in model.layers:
        layer.trainable = True

    # recompile
    model.compile(
        optimizer=Adam(lr=1.0e-4),
        loss=categorical_crossentropy,
        metrics=['accuracy'])

    # train the whole model
    hist2 = model.fit_generator(
        generator=generate_from_paths_and_labels(
            input_paths=train_input_paths,
            labels=train_labels,
            batch_size=batch_size_fine
        ),
        steps_per_epoch=math.ceil(len(train_input_paths) / batch_size_fine),
        epochs=epochs_fine,
        validation_data=generate_from_paths_and_labels(
            input_paths=val_input_paths,
            labels=val_labels,
            batch_size=batch_size_fine
        ),
        validation_steps=math.ceil(len(val_input_paths) / batch_size_fine),
        verbose=1,
    )


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
    d = datetime.datetime.today()
    dirname = 'result_%s%s%s%s%s' % (d.year, d.month, d.day, d.hour, d.minute)
    result_path = os.path.join(result_root, dirname)
    if os.path.exists(result_path) == False:
        os.makedirs(result_path)

    plt.plot(range(epochs), acc, marker='.', label='acc')
    plt.plot(range(epochs), val_acc, marker='.', label='val_acc')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.savefig(os.path.join(result_path, 'acc.png'))
    plt.clf()

    plt.plot(range(epochs), loss, marker='.', label='loss')
    plt.plot(range(epochs), val_loss, marker='.', label='val_loss')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(result_path, 'loss.png'))
    plt.clf()

    # save plot data as pickle file
    plot = {
        'acc': acc,
        'val_acc': val_acc,
        'loss': loss,
        'val_loss': val_loss,
    }
    with open(os.path.join(result_path, 'plot.dump'), 'wb') as f:
        pkl.dump(plot, f)

    # save model
    model.save(os.path.join(result_path, 'model.h5'))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
