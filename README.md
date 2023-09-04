# DOA-USING-image-translation

## Introduction

This project is a reproduction of "Sound source localization based on multi-task learning and image translation network", (since the original text did not give the reproduction code, the reproduction result is not good, there may be some parts that are not directly mentioned in the paper content).

## Ini-process
```
python3 doa_process.py
```
In this step, the code will generate input images,target 1 images and target 2 images.

The process may take a lot of time.

## Trianing 
```
python3 train.py
```
In this step,The code will use the network structure mentioned in the article for training, and the default setting parameters are consistent with the paper.

Parameters will be stored in weights_per_epoch.
