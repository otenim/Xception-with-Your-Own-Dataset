from keras.applications.xception import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
import numpy as np
import os
import argparse
import random

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
parser.add_argument(
    '--num_train',
    type=int,
    default=4000,
)
parser.add_argument(
    '--num_test',
    type=int,
    default=400,
)
parser.add_argument(
    '--num_classes',
    type=int,
    default=102,
)

def main(args):

    num_classes = args.num_classes
    num_train = args.num_train
    num_test = args.num_test
    # prepare data
    # train images and labels: 7281
    # test images and labels: 1864
    train = open(args.train_path, 'r')
    test = open(args.test_path, 'r')
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for line in train.readlines():
        line = line.strip()
        image, label = line.split()
        x_train.append(image)
        y_train.append(int(label))
    for line in test.readlines():
        line = line.strip()
        image, label = line.split()
        x_test.append(image)
        y_test.append(int(label))

    # shuffle data
    index_train = list(range(len(x_train)))
    index_test = list(range(len(x_test)))
    random.shuffle(index_train)
    random.shuffle(index_test)
    x_train_shuffled = []
    y_train_shuffled = []
    x_test_shuffled = []
    y_test_shuffled = []
    for i in index_train:
        x_train_shuffled.append(x_train[i])
        y_train_shuffled.append(y_train[i])
    for i in index_test:
        x_test_shuffled.append(x_test[i])
        y_test_shuffled.append(y_test[i])
    del x_train
    del y_train
    del x_test
    del y_test
    train.close()
    test.close()

    # make numpy array
    x_train = [img_to_array(load_img(path)) for path in x_train_shuffled[:num_train]]
    y_train = [y for y in y_train_shuffled[:num_train]]
    x_test = [img_to_array(load_img(path)) for path in x_test_shuffled[:num_test]]
    y_test = [y for y in y_test_shuffled[:num_test]]

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    print("%d train images were loaded" % len(x_train))
    print("%d train labels were loaded" % len(y_train))
    print("%d test images were loaded" % len(x_test))
    print("%d test labels were loaded" % len(y_test))

    # preprocess image data
    print("preprocessing image data...")
    x_train = preprocess_input(x_train)
    x_test = preprocess_input(x_test)

    # Convert class vectors to binary class matrices
    print("convert label vector to categorical one...")
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)


    print("Saving numpy array...")
    np.save("x_train", x_train)
    np.save("y_train", y_train)
    np.save("x_test", x_test)
    np.save("y_test", y_test)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
