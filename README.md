# Training Xception with your own dataset


## Description


This repository contains some scripts to train [Xception](https://arxiv.org/pdf/1610.02357.pdf) introduced by François Chollet, the author of Keras, a popular deeplearning framework.


## Environments


* GTX1070 (8GB) A middle-range or more powerful GPU is required.
* python 3.5.\*-3.6.\*
* numpy 1.13.\*-1.14.\*
* scipy 0.19.\*-1.0.\*
* h5py 2.6.\*-2.7.\*
* Keras 2.0.\*-2.1.\*
* tensorflow-gpu 1.4.\*-1.8.\*

If your environment is different from the above one, we can not guarantee our scripts correctly work.


## Demo


In this demonstration, we will train Xception on the caltech101 dataset (9145 images, 102 classes) as an example.  


#### 1. Prepare dataset


First, donwload and expand the dataset with the following command.  

```bash
$ sh download_dataset.sh
$ tar zxvf 101_ObjectCategories.tar.gz
```


#### 2. Make classes.txt


You must create a text file which lists all the class names.
It can be done with the following command.

```bash
$ ls 101_ObjectCategories > classes.txt
```


#### 3. Train the model


```bash
$ python fine_tune.py 101_ObjectCategories/ classes.txt ./result
```

In fine\_tune.py...  

* Pre-trained weights on ImageNet provided by official keras team are used as Xception's initial weights.  
* We first train only the top classifier for 5 epochs.
* Then, retrain the whole model for 50 epochs with lower learning rate.
* All the result data (model snapshot h5 format files, graphs) will be saved in `./result`.


#### 4. Inference


```bash
$ python inference.py ./result/model_fine_final.h5 classes.txt images/airplane.jpg
```

Input image:

![image](https://i.imgur.com/AsesiD0.jpg)  

Output result:

![result](https://i.imgur.com/5GeXqgl.png)


## How to train with your own dataset ?


### What you have to prepare


#### 1. A dataset you want to learn


You have to prepare a directory which is **the same format as the caltech101 dataset** as shown bellow:  

![Imgur](http://i.imgur.com/qBa9cKr.png)  

As an example, the above dataset has 3 classes and 5 images in total.  
The image file names can be anything.


#### 2. classes.txt


You have to create a text file which lists all the class names in each line.  
It can be created with the following command.

```bash
$ ls root/ > classes.txt
```

The file name needs not to be 'classes.txt', but can be anything.


### Let's train your model on your own dataset !!


```bash
$ python fine_tune.py root/ classes.txt <result_root> [epochs_pre] [epochs_fine] [batch_size_pre] [batch_size_fine] [lr_pre] [lr_fine] [snapshot_period_pre] [snapshot_period_fine]
```
NOTE: [] indicates an optional argument. <> indicates a required argument.

* `<result_root>`: Path to the directory where all the result data will be saved.
* `[epochs_pre]`: The number of epochs during training only the top classifier (default:5),
* `[epochs_fine]`: The number of epochs during training the whole model (default:50).
* `[batch_size_pre]`: Size of a mini-batch during training only the top classifier (default:32).
* `[batch_size_fine]`: Size of a mini-batch during training the whole model (default:16).
* `[lr_pre]`: Learning rate during training only the top classifier (default:1e-3)
* `[lr_fine]`: Learning rate during training the whole model (default:1e-4)
* `[snapshot_period_pre]`: Snapshot period during training only the top classifier (default:1). At the every spedified number of epochs, a model snapshot h5 format file will be saved into <result_root>.
* `[snapshot_period_fine]:`: Snapshot period during training the whole model (default:1).

For example, if you'd like to pre-train a model for 2 epochs and finetune it for 10 epochs with learning rate 5e-4, then save all the result data in `./result`, please run the following command.

```bash
$ python fine_tune.py root/ classes.txt ./result --epochs_pre 2 --epochs_fine 10 --lr_fine 5e-4
```


## How to inference with your trained model ?


```bash
$ python inference.py <model> <classes> <image> [top_n]
```
NOTE: [] indicates an optional argument. <> indicates a required argument.

* `<model>`: Path to a keras model h5 format file.
* `<classes>`: Path to a txt file where all the class names are listed at each line.
* `<image>`: Path to an image file which is to be classied.
* `[top_n]`: Show top n results (default: 10).


## Future work

* Expand our scripts to be able to train other popular classification models (for example, VGG16, VGG19, MobileNet, DenseNet, InceptionV3, etc.) without breaking changes.
