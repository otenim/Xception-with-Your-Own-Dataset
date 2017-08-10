Training Xception with your own dataset
====================================

## Description  
This repository contains some scripts to train Xception devised by François Chollet, the author of keras which is a popular machine learning framework.  

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
#### 1. A dataset you want to learning
You have to prepare a directory similar to the format of caltech101
as shown bellow:  
#### 2. classes.txt
### How to tune the hyper parameters ?
![Imgur](http://i.imgur.com/qBa9cKr.png)  
