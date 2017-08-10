#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import random
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.xception import preprocess_input
from keras.utils import to_categorical

current_dir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('root_dir')
parser.add_argument('--split', type=float, default=0.8)
parser.add_argument('--classes', default=os.path.join(current_dir, 'classes.txt'))

def main(args):
    root_dir = os.path.abspath(args.root_dir)

    # make class list
    classes = []
    f_classes = open(args.classes, 'r')
    classes = list(map(lambda x: x.strip(),f_classes.readlines()))
    f_classes.close()

    # make image_paths and labels
    sum_img = 0
    image_paths = []
    labels = []
    for class_dir in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_dir)
        class_name = class_path.split('/')[-1]

        print("========== processing on class (%s) ==========" % (class_name))
        sum_class_img = 0
        label = classes.index(class_name)
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            image_paths.append(img_path) # add an image path
            labels.append(label) # add a label
            sum_class_img += 1
            sum_img += 1
        print("%d image paths were loaded." % (sum_class_img))
    print("========== result ==========")
    print("%d classes, %d images were loaded in total" % (len(classes), sum_img))
    print("============================")

    # shuffle image_paths and labels
    index = list(range(len(image_paths)))
    random.shuffle(index)
    image_paths_shuffled = []
    labels_shuffled = []
    for i in index:
        image_paths_shuffled.append(image_paths[i])
        labels_shuffled.append(labels[i])
    del image_paths
    del labels

    # make numpy arrays
    print("converting image paths and labels into requsite numpy arrays...")
    border = int(len(image_paths_shuffled) * args.split)
    x_train = [img_to_array(load_img(path)) for path in image_paths_shuffled[:border]]
    y_train = labels_shuffled[:border]
    x_test = [img_to_array(load_img(path)) for path in image_paths_shuffled[border:]]
    y_test = labels_shuffled[border:]
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    print("train: %d images and labels." % (len(x_train)))
    print("test: %d images and labels." % (len(x_test)))
    del image_paths_shuffled
    del labels_shuffled

    # preprocess image data
    print("preprocessing image data...")
    x_train = preprocess_input(x_train)
    x_test = preprocess_input(x_test)

    # Convert class vectors to binary class matrices
    print("convert label vector to categorical one...")
    y_train = to_categorical(y_train, len(classes))
    y_test = to_categorical(y_test, len(classes))

    # save numpy arrays
    print("Saving numpy array...")
    if not('dataset' in os.listdir(current_dir)):
        os.mkdir(os.path.join(current_dir, 'dataset'))
    np.save(os.path.join(current_dir, 'dataset', 'x_train'), x_train)
    np.save(os.path.join(current_dir, 'dataset', 'y_train'), y_train)
    np.save(os.path.join(current_dir, 'dataset', 'x_test'), x_test)
    np.save(os.path.join(current_dir, 'dataset', 'y_test'), y_test)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
