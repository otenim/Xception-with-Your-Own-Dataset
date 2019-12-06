# Training Xception with your own dataset

## Description

This repository contains some scripts to train [Xception](https://arxiv.org/pdf/1610.02357.pdf) introduced by François Chollet, the founder of Keras.

## Environments

We tested our scripts in the following environment.

* GTX1070 (8GB) A middle-range or more powerful GPU is required.
* python 3.5.\*-3.6.\*
* numpy 1.13.\*-1.17.\*
* scipy 0.19.\*-1.3.\*
* h5py 2.6.\*-2.10.\*
* Keras 2.0.\*-2.3.\*
* tensorflow-gpu 1.4.\*-1.15.\*

## Demo

Here, we'll show how to train Xception on the Caltech101 dataset (9145 images, 102 classes) as an example.

#### 1. Prepare dataset

Please download and expand the dataset with the following command.  

```bash
$ sh download_dataset.sh
$ tar zxvf 101_ObjectCategories.tar.gz
```

#### 2. Make classes.txt

You must create a text file where all the class names are listed line by line.  
This can be easily done with the below command.

```bash
$ ls 101_ObjectCategories > classes.txt
```

#### 3. Train the model

```bash
$ python fine_tune.py 101_ObjectCategories/ classes.txt result/
```

In `fine_tune.py`...  

* Xception's weights are initialized with the ones pre-trained on the ImageNet dataset (officialy provided by the keras team).
* In the first training stage, only the top classifier of the model is trained for 5 epochs.
* In the second training stage, the whole model is trained for 50 epochs with a lower learning rate.
* All the result data (serialized model files and figures) are to be saved under `result/`

#### 4. Inference

```bash
$ python inference.py result/model_fine_final.h5 classes.txt images/airplane.jpg
```

**[Input Image]**:  
![image](https://i.imgur.com/AsesiD0.jpg)  

**[Output Result]**:  
![result](https://i.imgur.com/5GeXqgl.png)

## How to train with your own dataset ?

### What do you have to prepare ?

#### 1. A dataset you wanna use

You have to prepare a directory which has the same structure as the caltech101 dataset
as shown bellow:  
![Imgur](http://i.imgur.com/qBa9cKr.png)

The above example dataset has 3 classes and 5 images in total.
Each class name must be unique, but the image files' can be anything.

#### 2. classes.txt

You have to create a text file where all the class names are listed line by line. This can be done with the following command.

```bash
$ ls root/ > classes.txt
```

The file name does not need to be `classes.txt`, but
you can name it as you want.

### Let's train your model on your own dataset !!

```bash
$ python fine_tune.py root/ classes.txt <result_root> [epochs_pre] [epochs_fine] [batch_size_pre] [batch_size_fine] [lr_pre] [lr_fine] [snapshot_period_pre] [snapshot_period_fine]
```
NOTE: [] indicates an optional argument. <> indicates a required argument.

* `<result_root>`: Path to the directory where all the result data will be saved.
* `[epochs_pre]`: The number of epochs during the first training stage (default: 5).
* `[epochs_fine]`: The number of epochs during the second training stage (default: 50).
* `[batch_size_pre]`: Batch size during the first training stage (default: 32).
* `[batch_size_fine]`: Batch size during the second training stage (default: 16).
* `[lr_pre]`: Learning rate during the first training stage (default:1e-3).
* `[lr_fine]`: Learning rate during the second training stage (default:1e-4).
* `[snapshot_period_pre]`: Snapshot period during the first training stage (default:1). At the every spedified epochs, a serialized model file will be saved under <result_root>.
* `[snapshot_period_fine]`: Snapshot period during the second training stage (default:1).

For example, if you'd like to pre-train a model for 2 epochs with leraning rate 5e-3 and fine-tune it for 10 epochs with learning rate 5e-4, please run the following command.

```bash
$ python fine_tune.py root/ classes.txt result/ --epochs_pre 2 --epochs_fine 10 --lr_pre 5e-3 --lr_fine 5e-4
```

## How to inference with your trained model ?

```bash
$ python inference.py <model> <classes> <image> [top_n]
```
NOTE: [] indicates an optional argument. <> indicates a required argument.

* `<model>`: Path to a serialized model file.
* `<classes>`: Path to a txt file where all the class names are listed line by line.
* `<image>`: Path to an image file that you would like to classify.
* `[top_n]`: Show top n results (default: 10).

## Future work

* Expand our scripts to be able to train other popular classification models (for example, VGG16, VGG19, MobileNet, DenseNet, InceptionV3, etc.) without breaking changes.
