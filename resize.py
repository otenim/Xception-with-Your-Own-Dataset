#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
from PIL import Image
import numpy as np
import glob

current_dir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('src_path')
parser.add_argument('--width', type=int, default=299)
parser.add_argument('--height', type=int, default=299)

def main(args):
    src_path = os.path.abspath(args.src_path)
    target_size = (args.width, args.height)

    # make distination directory
    src_dir_name = src_path.split('/')[-1]
    dst_dir_name = src_dir_name + '_resized'
    dst_path = os.path.join(current_dir, dst_dir_name)
    if os.path.exists(dst_path) == False:
        os.mkdir(dst_path)

    # resize all images into new_size(width, height)
    for class_dir in os.listdir(src_path):
        class_path = os.path.join(src_path, class_dir)
        class_name = class_path.split('/')[-1]
        print("========== class_name: %s ==========" % (class_name))

        # make subdirecotry unuder dst_path
        dst_class_path = os.path.join(dst_path, class_dir)
        if os.path.exists(dst_class_path) == False:
            os.mkdir(dst_class_path)

        # open all images in dst_class_path and resize to target_size
        for img_file in glob.glob(os.path.join(class_path, '*.jpg')):
            img_path = os.path.join(class_path, img_file)
            dst_img_path = os.path.join(dst_class_path, img_file)
            img = Image.open(img_path).convert('RGB')
            img_resized = img.resize(target_size)
            img_resized.save(dst_img_path, 'JPEG')
            print("for %s: resized %s => %s" % (img_path, img.size, img_resized.size))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
