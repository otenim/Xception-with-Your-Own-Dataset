from keras.applications.xception import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
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
    print("%d train images were loaded" % len(x_train))
    print("%d train labels were loaded" % len(y_train))
    print("%d test images were loaded" % len(x_test))
    print("%d test labels were loaded" % len(y_test))
    train.close()
    test.close()


    x_train = np.asarray([img_to_array((load_img(path))) for path in x_train])
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

    # shuffle
    np.random.shuffle(x_train)
    np.random.shuffle(y_train)
    np.random.shuffle(x_test)
    np.random.shuffle(y_test)


    print("Saving numpy array...")
    np.save("x_train", x_train[:4000])
    np.save("y_train", y_train[:4000])
    np.save("x_test", x_test[:800])
    np.save("y_test", y_test[:800])


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
