Training Xception with your own dataset
====================================

## Description  
This repository contains some scripts to train Xception devised by François Chollet, the author of keras which is a popular machine learning framework.  

## Environment
* GTX1070(8GB) A powerful GPU is required.
* python 3.5.2
* numpy 1.13.1
* scipy 0.19.1
* h5py 2.6.0
* keras 2.0.6

## Demo
In the demonstration, we train Xception with the dataset of caltech101
(9145 images, 102 classes) as an example.  

#### 1. Preparing dataset
First, donwload and expand the dataset with the following command.  
`sh download_dataset.sh && tar xvf 101_ObjectCategories.tar.gz`  

Second, resize the all images with the size (width, height) = (299, 299).  
`python resize.py 101_ObjectCategories/`

You'll get the resized dataset whose name is '101_ObjectCategories_resized'.  

#### 2. Create requsite numpy arrays
Create the resusite numpy arrays with the following command.  
`python create_dataset.py 101_ObjectCategories_resized/`  

Then, you'll get 'dataset' directory which contains
x_train.npy, y_train.npy, x_test.npy, and y_test.npy  

#### 3. Train the model
Training will start just by executing the following command.  
`python fine_tune.py`  

In fine_tune.py, imagenet's weight is used as an initial weight of Xception.  
We first train only the top of the model(Classifier) for 10 epochs, and
then retrain the whole model for 200 epochs with lower learning rate.  

When the training ends, 'result' directory is to be created.  
This directory contains 2 graphs(loss.png and acc.png) which shows the
training results and 1 dump file which is consisted of the plot data.


## How to train with my own dataset ?
### What you have to prepare
#### 1. A dataset you want to learn
You have to prepare a directory which is similar to the format of caltech101
as shown bellow:  
![Imgur](http://i.imgur.com/qBa9cKr.png)  
As an example, this dataset has 3 classes and 5 images in total.  
The name of the image file in the class directory can be anything.  

#### 2. classes.txt
You must prepare a text file that lists all class names.  
It's very easy to make this file, I made it with a command like the following.  
`ls root/ >> classes.txt`  

Note: Here we have the name 'classes.txt' as an example, but in fact the name can be anything.  

### Let's train with your own dataset
First, create requsite numpy arrays  
`python resize.py root/ --width=299 --height=299`  
`python create_dataset.py root_resized/ --classes=<path_to_classes.txt> --split=0.8`  

4 numpy arrays(x_train.npy, y_train.npy, x_test.npy and y_test.npy) will
be generated inthe 'dataset' directory.

Second, train Xception  
`python fine_tune.py --epochs_pre=10 --epochs_fine=200
--batch_size_pre=32 --batch_size_fine=16  --classes=<path_to_classes.txt>
--dataset_path=<path_to_dataset>`  
