# Deep stereo matching using Tensorflow

This is a TensorFlow implementation of the stereo matching algorithm described in the paper ["Efficient Deep Learning for Stereo Matching"](https://www.cs.toronto.edu/~urtasun/publications/luo_etal_cvpr16.pdf). 

## Compatibility
The code is tested using Tensorflow r1.4 under Ubuntu 14.04 with Python 2.7 and Python 3.5.

## Training data
The [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) dataset has been used for training. This dataset consists of total of 200 scenes for training and of 200 scenes for testing. For more details, please check the KITTI website.

## Pre-processing
For training and validation, locations from the ground truth disparity images are generated using the preprocessing scripts published in the paper code available [here](https://bitbucket.org/saakuraa/cvpr16_stereo_public/src/1a41996ef7dda999b43d249fd51442d0b2e9dd0f/preprocess/?at=master).
This preprocessing script generates 3 binary files. Only the training and validation binary files are used.

## Running training
You can run the training as follows : 

```
python train_sm.py --data_dir data_path \
--log_dir log_dir \
--train_loc train_binary_file\
--valid_loc validation_binary_file &
```
You can skip validation by removing the "valid_loc" argument.

## Performance on test images
To generate stereo estimates using the pretrained model, this can be done using 

```
python validate_on_test_images.py --data_dir test_data_path \
--model mode_dir \
```
This script will generate stereo estimates for the whole testing images set.

