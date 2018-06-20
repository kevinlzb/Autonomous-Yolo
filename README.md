# Yolo-v1/v2 Road Object Detector

This project is implemented from scratch based on Yolov1/v2 paper, for details please read [Yolov1](https://pjreddie.com/media/files/papers/yolo_1.pdf),
[Yolov2](https://pjreddie.com/media/files/papers/YOLO9000.pdf). This project is implemented in MXnet framework and trained by Nividia 
Geforece 1090. 

## Introduction
This project aims to detect 4 different classess of objects, which are car, traffic lights, pedestrian and cyclist. The input for yolov1 and yolov2 are 224 * 224 and 416 * 416 repectively.
The output of the Yolov1 model is -1 * 7 * 7 * 5 and for Yolov2 is -1 * 13 * 13 * 5 * 9. K-means algorithm is used to calculate the anchor boxes for yolov2.

## Result
The network is trained on Nvidia Geforce and is trained for 600 epoches and 30 epoches with dataset 10k and 130k repectively. The accuracy and precsion are easier to train, but width and height are difficult to learn,
there is around 1-pixel off for the dimension of object. Here I applied my network on some random images. 
<img src="https://github.com/kevinlzb/Autonomous-Yolo/blob/master/result/3.PNG" alt="img1"/>
<img src="https://github.com/kevinlzb/Autonomous-Yolo/blob/master/result/1.PNG" alt="img2"/>


## Requirements
1.You can trained on your own GPU or use [AWS Instance](https://aws.amazon.com/marketplace/pp/B01M0AXXQ)
2.Install opencv
3.Install [python](https://www.python.org/downloads/)
4. Install MXNet 
Please goes to [Install MXNet](https://mxnet.incubator.apache.org/install/index.html?platform=Linux&language=Python&processor=CPU)
### prerequirement
    sudo apt-get update
    sudo apt-get install -y wget python gcc
    wget https://bootstrap.pypa.io/get-pip.py && sudo python get-pip.py
### install mxnet
    pip install mxnet
### install graphviz
    sudo apt-get install graphviz
    pip install graphviz

## Preparation for Training
1. please download Resnet for the feature extraction [Resnet](https://github.com/tornadomeet/ResNet)
2. There are 10k images in the dataset in the beginning, but beacause the network is a litte bit overffitng so I did a data augmentation to increase the dataset to 130k finally.
   Run [data_augmention] to increase the dataset by adjust the brightness x2, blur x2, constract x2 for each image.
3. Run [data_preprocess] file to convert image to .rec file which is used for large dataset trainning in MXNet.

## Future improvements
1. In the implementation of yolov2, I simpy used comman K-means algorithm to find 5 anchor boxes, but it is better to think about that box with larger size may 
have more influnce on erroes.
2. Continue to increase dataset, also increase the objectness to more than 4.
3. Read yolov3.
