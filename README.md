Training Xception with your dataset
====================================

## Description  
This repository contains some scripts to train Xception devised by François Chollet, the author of keras which is a popular machine learning framework.  

## Demo
In the demonstration, we train Xception with the dataset of caltech101
(9145 images, 102 classes) as an example.  

#### 1. Preparing dataset
First, donwload and expand the dataset with the following command.  
`sh download_dataset.sh && tar xvf 101_ObjectCategories/`  

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

## How to train with my dataset ?  
Download caltech101 dataset and expand it.  
`sh download_dataset.sh && tar xvf 101_ObjectCategories.tar.gz`  

データセットを(width, height) = (299, 299)にリサイズ  
`python resize.py 101_ObjectCategories/`  
リサイズが完了すると,101_ObjectCategories_resizedというディレクトリが生成される.  

学習に必要なnpyファイルを作成  
`python create_dataset.py 101_ObjectCategories_resized`  

## 学習

学習させたいネットワークのディレクトリに移動(例:inceptionv3)し,  
`cd inceptionv3`  

学習をする.  
`python fine_tune.py [--epochs_pre] [--epochs_fine] [--batch_size_pre] [--batch_size_fine`]  
学習の終了後にはinceptionv3/に比較グラフ作成用のdumpファイルが出力される.  

両方のネットワークを一気に学習させたい時は,ルートに移動してから  
`python xception/fine_tune.py && python inceptionv3/fine_tune.py`  
とやると非常に楽.  

### オプションの説明
epochs_pre: 分類器の学習フェーズのエポック数(初期値:10)  
epochs_fine: ネットワーク全体の学習フェーズのエポック数(初期値:200)  
batch_size_pre: 分類器の学習フェーズのバッチサイズ(初期値:32)  
batch_size_fine: ネットワーク全体の学習フェーズのバッチサイズ(初期値:16)  

## 比較グラフの作成
inceptionv3とxceptionの学習が終わり,それぞれのdumpファイルが生成されていると仮定する.  
ルートに移動し,  
`python make_graph.py`  
とすると,ルートに損失と精度を比較したloss.pngとacc.pngが生成される.  

`python xception/fine_tune.py && python inceptionv3/fine_tune.py && python make_graph.py`  
とすると,両ネットワークの学習からグラフの作成までを1コマンドで処理することが可能.  
