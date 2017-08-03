#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('root_dir')
parser.add_argument('--classes', default=os.path.join(current_dir, 'classes.txt'))

def main(args):
    root_dir = os.path.abspath(args.root_dir)

    # make classes
    classes = []
    f_classes = open(args.classes, 'r')
    classes = list(map(lambda x: x.strip(),f_classes.readlines()))
    f_classes.close()

    # make train.txt test.txt
    f_train = open(os.path.join(current_dir, 'train.txt'), 'w')
    f_test = open(os.path.join(current_dir, 'test.txt'), 'w')
    split_ratio = 0.8

    sum_img = 0

    for class_dir in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_dir)
        class_name = class_path.split('/')[-1]
        image_paths = []
        print("========== class name: %s ==========" % (class_name))
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            image_paths.append(img_path)
            sum_img += 1

        # split image_paths for train and test
        border = int(len(image_paths) * split_ratio)
        train_image_paths = image_paths[:border]
        test_image_paths = image_paths[border:]
        print("train: %d images" % (len(train_image_paths)))
        print("test: %d images" % (len(test_image_paths)))

        # get class id
        class_id = classes.index(class_name)

        # write image paths and class ids to train.txt and test.txt
        train_image_paths = list(map(lambda x: x+' '+str(class_id)+'\n', train_image_paths))
        test_image_paths = list(map(lambda x: x+' '+str(class_id)+'\n', test_image_paths))
        f_train.writelines(train_image_paths)
        f_test.writelines(test_image_paths)

    f_train.close()
    f_test.close()
    print("%d classes, %d images were loaded" % (len(classes), sum_img))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
