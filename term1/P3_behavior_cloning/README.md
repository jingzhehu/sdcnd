# P3 Behavior Cloning

## Solution videos

The solution for [track 1](https://youtu.be/57bOB77L498) and [track 2](https://youtu.be/JnvnS5FE2_A) are published on YouTube. The settings used for Udacity car simulator were: 1) Graphics: `640 x 480`, 2) Graphics quality: `fast` and 3) `windowed`. Regarding hardware, a `Nvidia 980Ti` was used for both training and inference. Please use the `drive.py` supplied in this repository to reproduce the results.  

## Training data

Network training proceeded in **two stages**: 1) **Udacity data** (only track one) and 2) training with **live-trainer**. 
The initial training set is composed of car simulator center camera images and steering angle provided by Udacity (`8036 samples`). 

### Udacity data
Each image is read in as `uint8`, resized to `(66, 200)`, converted from `RGB` to `HLS` color space and finally casted as `float32`. The image is then standardized by `img/255.0 - 0.5`.

The training set needs to be balanced since around 50% of the steering angles associated with the images are zero. In this case, the steering angles are divided into `20 bins` each with `2387 samples` (4774 samples in the bin containing "0"). After applying random translation and rotations, the augmented training image dataset has a final shape of `(52515, 66, 200, 3, dtype=float32)`. The details can be found in the accompanying iPython notebook (`P3_nv_xception.ipynb`) or HTML export.

### Live-trainer 
Fellow udacian, **Thomas Antony** created a [live trainer](https://github.com/thomasantony/sdc-live-trainer) for the behavior cloning project, which in turn is inspired by **John Chen**'s [Agile Trainer](https://github.com/diyjac/AgileTrainer).  Building upon their work, I implemented a rudimentary data recording feature and changed the image preprocessing part. With the live-trainer, it's possible to toggle between autonomous mode and training mode on the fly, thus focusing on training the network on the difficult parts of the track only. 

In the end, I was able to train the network to successfully navigate both the tracks (unfortunately using two sets of weights). It might be possible to use a single set of weights to navigate two tracks, but that might require considerably more efforts in selecting the relevant training data.

### Possible improvements
It might be advantageous to use lane finding techniques to augment training data in order to more effectively train the network. One possibility is to detect the two lane marks and synthesize a median. The network trained would then learn to respond to deviation from the median in addition to recovery from approaching the side lane marks. The end result might be more steady steering angle outputs. 


## Network architecture

The **nv_xception model** borrows the general structure from NVIDIA self-driving car paper ([Bojarski et al](https://arxiv.org/abs/1604.07316))
and replaces several convolution layers with depth-wise separable convolution layer ([Chollet](https://arxiv.org/abs/1610.02357)).

The resulting model accepts `input` image data tensors of shape `(None, 66, 200, 3)` followed by 5 blocks. Each network block generally contains a convolution layer, a batch normalization layer and a relu activation layer. Blocks 1-2 uses 2D convolution
layers followed by a max pooling layer while blocks 3-5 opt for depth-wise convolution layer. After average pooling, the results
would pass through a dropout layer before finally connected to the `output` dense layer `(None, 1)`. The optimizer used was `RMSprops` with the initial learning rate set to `0.008`, which is chosen on the conservative side and a bit less than the default setting of `0.01` in Keras. In this case, I elected not to experiment too much with different learning rates using the Udacity data since the mean squared error (MSE) metric might not be indicative of actual on-road performance. Instead, I decided to place more emphasis on tuning the network with `live-trainer`, which helped to produce the desired results more efficiently. The network details are listed below.

|Layer (type)                     |Output Shape          |Param #     |Connected to|
|---------------------------------|----------------------|------------|------------|
|input_layer (InputLayer)         |(None, 66, 200, 3)    |0           |            |
|block1_conv (Convolution2D)      |(None, 62, 196, 48)   |3648        |standardize_imgs[0][0]|
|block1_bn (BatchNormalization)   |(None, 62, 196, 48)   |192         |block1_conv[0][0]|
|block1_act (Activation)          |(None, 62, 196, 48)   |0           |block1_bn[0][0]|
|block2_conv (Convolution2D)      |(None, 58, 192, 96)   |115296      |block1_act[0][0]|
|block2_bn (BatchNormalization)   |(None, 58, 192, 96)   |384         |block2_conv[0][0]|
|block2_act (Activation)          |(None, 58, 192, 96)   |0           |block2_bn[0][0]|
|block2_max_pool (MaxPooling2D)   |(None, 29, 96, 96)    |0           |block2_act[0][0]|
|block3_sepconv (SeparableConvolu |(None, 25, 92, 192)   |21024       |block2_max_pool[0][0]|
|block3_bn (BatchNormalization)   |(None, 25, 92, 192)   |768         |block3_sepconv[0][0]|
|block3_act (Activation)          |(None, 25, 92, 192)   |0           |block3_bn[0][0]|
|block3_max_pool (MaxPooling2D)   |(None, 12, 46, 192)   |0           |block3_act[0][0]|
|block4_sepconv (SeparableConvolu |(None, 10, 44, 384)   |75840       |block3_max_pool[0][0]|
|block4_bn (BatchNormalization)   |(None, 10, 44, 384)   |1536        |block4_sepconv[0][0]|
|block4_act (Activation)          |(None, 10, 44, 384)   |0           |block4_bn[0][0]|
|block5_sepconv (SeparableConvolu |(None, 8, 42, 768)    |299136      |block4_act[0][0]|
|block5_bn (BatchNormalization)   |(None, 8, 42, 768)    |3072        |block5_sepconv[0][0]|
|block5_act (Activation)          |(None, 8, 42, 768)    |0           |block5_bn[0][|0]|
|avg_pool (GlobalAveragePooling2D |(None, 768)           |0           |block5_act[0][|0]|
|drop_out |(Dropout)               |(None, 768)           |0           |avg_pool[0][0]|
|steering_angle (Dense)           |(None, 1)             769         drop_out|[0][0]

>Total params: 521,665
>
>Trainable params: 518,689
>
>Non-trainable params: 2,976

