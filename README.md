# DeepHash-Paddle
DeepHash research based on paddle

## Environment
Paddle 2.1.0 python3.7

## Dataset
* cifar10
* cifar10-1
* cifar10-2
* imagenet100
* nuswide_21
* nuswide_21_m
* nuswide_81_m
* voc2012
* coco
* mirflickr

## How to Train
It is easy to train the deephash model, just run command as follows:
```
python DSHSD.py
```
The trained model will be save in *config["save_path"]* (by default: save/DSHSD)
