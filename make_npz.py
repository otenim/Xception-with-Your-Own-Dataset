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

    num_classes = 102
    # prepare data
    # train images and labels: 7281
    # test images and labels: 1864
    train = open(args.train_path, 'r')
    test = open(args.test_path, 'r')
    train_tmp = list(map(lambda x: x.strip(), train.readlines()))
    test_tmp = list(map(lambda x: x.strip(), test.readlines()))
    train_tmp = [line.split() for line in train_tmp]
    test_tmp = [line.split() for line in test_tmp]
    x_train = [x for (x, y) in train_tmp]
    y_train = [y for (x, y) in train_tmp]
    x_test = [x for (x, y) in test_tmp]
    y_test = [y for (x, y) in test_tmp]
    print("%d train images were loaded" % len(x_train))
    print("%d train labels were loaded" % len(y_train))
    print("%d test images were loaded" % len(x_test))
    print("%d test labels were loaded" % len(y_test))

    x_train = np.asarray([img_to_array(load_img(path)) for path in x_train])
    y_train = np.asarray([y for y in y_train])
    x_test = np.asarray([img_to_array(load_img(path)) for path in x_test])
    y_test = np.asarray([y for y in y_test])

    # preprocess image data
    print("preprocessing image data...")
    x_train = preprocess_input(x_train)
    x_test = preprocess_input(x_test)

    # Convert class vectors to binary class matrices
    print("convert label vector to categorical one...")
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # save numpy array
    perm_train = np.random.permutation(len(x_train))
    perm_test = np.random.permutation(len(x_test))
    x_train = x_train[perm_train]
    y_train = y_train[perm_train]
    x_test = x_test[perm_test]
    y_test = y_test[perm_test]

    print("Saving numpt array...")
    np.save("x_train", x_train[:4000])
    np.save("y_train", y_train[:4000])
    np.save("x_test", x_test[:1000])
    np.save("y_test", y_test[:1000])


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
