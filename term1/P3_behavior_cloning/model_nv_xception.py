import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

from keras import callbacks
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils.visualize_util import plot
from keras.layers import Activation, BatchNormalization, Convolution2D, Dropout
from keras.layers import Dense, Input, SeparableConvolution2D, MaxPooling2D, GlobalAveragePooling2D, Lambda

# parameters
IMG_DIR = os.path.join("training_data", "IMG_udacity/")
DRIVE_LOG = os.path.join("training_data", "driving_log.csv")
INPUT_SHAPE = (66, 200, 3)
TEST_SIZE = 0.15

EPOCHS = 15
BATCH_SIZE = 32
L_RATE = 0.008
DROPOUT_P = 0.5
MODEL_NAME = 'nv_xception_model.json'
MODEL_WEIGHT_NAME = 'nv_xception_model.h5'
MODEL_GRAPH_NAME = 'nv_xception_model.png'
LOAD_WEIGHTS = False


def preprocess_input(img):
    '''Corp and resize images'''
    img = cv2.resize(img[60:140, 40:280], (200, 66))

    return np.array(img, dtype="uint8")


def class_indices(input_array):
    '''Return a list of indices for each class (modified from https://goo.gl/abAOe6)'''

    idx_sort = np.argsort(input_array)
    input_array_sorted = input_array[idx_sort]
    vals, idx_start, count = np.unique(input_array_sorted, return_counts=True, return_index=True)

    return np.split(idx_sort, idx_start[1:])


def augment_image_single(img, ang_range=5, shear_range=0, trans_range=10):
    '''Return an img with random rotation, translation and shear

    Modified from Yivek Yadav's approach using OpenCV (https://goo.gl/ttRKL0)
    '''

    img_aug = np.copy(img)
    height, width, ch = img_aug.shape

    # rotation
    if ang_range != 0:
        ang_rot = np.random.uniform(ang_range) - ang_range / 2
        rot_M = cv2.getRotationMatrix2D((width / 2, height / 2), ang_rot, 1)
        img_aug = cv2.warpAffine(img_aug, rot_M, (width, height))

    # translation
    if trans_range != 0:
        tr_x, tr_y = trans_range * np.random.uniform(size=(2, 1)) - trans_range / 2
        trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
        img_aug = cv2.warpAffine(img_aug, trans_M, (width, height))

    # shear
    if shear_range != 0:
        pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
        pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
        pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2
        pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
        shear_M = cv2.getAffineTransform(pts1, pts2)
        img_aug = cv2.warpAffine(img_aug, shear_M, (width, height))

    return img_aug


def augment_images(X_data, y_data, multiplier=1.5, y_bins=None):
    ''' Return an augmented image dataset (HLS color space) with same number of samples in each category / bin

    '''

    X_data_augs = [*X_data]
    y_data_augs = [*y_data]

    print(np.array(X_data_augs).shape)

    if y_bins is None:
        bin_indices_list = class_indices(y_data)
        bin_labels, bin_counts = np.unique(y_data, return_counts=True)
    else:
        y_data_bined = np.digitize(y_data, bins=y_bins)
        bin_indices_list = class_indices(y_data_bined)
        bin_labels, bin_counts = np.unique(y_data_bined, return_counts=True)

    n_target = np.int(multiplier * np.max(bin_counts))
    n_augs = [(n_target - bin_count) * ((n_target - bin_count) > 0) for bin_count in bin_counts]

    print("Target num of samples for each bin: {}".format(n_target))
    print("Total num of training samples: {}\n".format(len(y_data) + np.sum(n_augs)))
    for bin_label, n_aug, bin_indices_data in zip(bin_labels, n_augs, bin_indices_list):

        print("Augmenting bin: {:2} with {:4} samples".format(bin_label, n_aug))
        for idx, bin_idx in enumerate(np.random.choice(bin_indices_data, size=n_aug)):
            x_aug = augment_image_single(X_data[bin_idx])
            y_aug = y_data[bin_idx]

            # randomly flip the image horizontally
            flip = np.random.choice([True, False])
            if flip:
                x_aug = cv2.flip(x_aug, 1)
                y_aug = -y_aug

            X_data_augs.append(x_aug)
            y_data_augs.append(y_aug)

    # standardize inputs
    # Lambda layer in keras has serialization problems across machines
    X_data_augs = [cv2.cvtColor(x_aug, cv2.COLOR_RGB2HLS).astype("float32")/255.0 - 0.5 for x_aug in X_data_augs]

    return np.array(X_data_augs), np.hstack(y_data_augs)


def nv_xception(intputs):
    '''Return keras model

    The nv_xception model borrows the general structure from NVIDIA self-driving car paper (Bojarski et al, https://arxiv.org/abs/1604.07316)
    and replaces several convolution layers with depth-wise separable convolution layer (Chollet, https://arxiv.org/abs/1610.02357).

    The resulting model accepts input image data tensors of shape (None, 66, 200, 3) followed by 5 blocks detailed below.
    Each block generally contains a conv layer, a batch normalization layer and a relu activation layer. Blocks 1-2 uses 2D convolution
    layers followed by a max pooling layer while blocks 3-5 opt for depth-wise convolution layer. After average pooling, the results
    would pass through a dropout layer before connected to the output dense layer.

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_layer (InputLayer)         (None, 66, 200, 3)    0
____________________________________________________________________________________________________
block1_conv (Convolution2D)      (None, 62, 196, 48)   3648        standardize_imgs[0][0]
____________________________________________________________________________________________________
block1_bn (BatchNormalization)   (None, 62, 196, 48)   192         block1_conv[0][0]
____________________________________________________________________________________________________
block1_act (Activation)          (None, 62, 196, 48)   0           block1_bn[0][0]
____________________________________________________________________________________________________
block2_conv (Convolution2D)      (None, 58, 192, 96)   115296      block1_act[0][0]
____________________________________________________________________________________________________
block2_bn (BatchNormalization)   (None, 58, 192, 96)   384         block2_conv[0][0]
____________________________________________________________________________________________________
block2_act (Activation)          (None, 58, 192, 96)   0           block2_bn[0][0]
____________________________________________________________________________________________________
block2_max_pool (MaxPooling2D)   (None, 29, 96, 96)    0           block2_act[0][0]
____________________________________________________________________________________________________
block3_sepconv (SeparableConvolu (None, 25, 92, 192)   21024       block2_max_pool[0][0]
____________________________________________________________________________________________________
block3_bn (BatchNormalization)   (None, 25, 92, 192)   768         block3_sepconv[0][0]
____________________________________________________________________________________________________
block3_act (Activation)          (None, 25, 92, 192)   0           block3_bn[0][0]
____________________________________________________________________________________________________
block3_max_pool (MaxPooling2D)   (None, 12, 46, 192)   0           block3_act[0][0]
____________________________________________________________________________________________________
block4_sepconv (SeparableConvolu (None, 10, 44, 384)   75840       block3_max_pool[0][0]
____________________________________________________________________________________________________
block4_bn (BatchNormalization)   (None, 10, 44, 384)   1536        block4_sepconv[0][0]
____________________________________________________________________________________________________
block4_act (Activation)          (None, 10, 44, 384)   0           block4_bn[0][0]
____________________________________________________________________________________________________
block5_sepconv (SeparableConvolu (None, 8, 42, 768)    299136      block4_act[0][0]
____________________________________________________________________________________________________
block5_bn (BatchNormalization)   (None, 8, 42, 768)    3072        block5_sepconv[0][0]
____________________________________________________________________________________________________
block5_act (Activation)          (None, 8, 42, 768)    0           block5_bn[0][0]
____________________________________________________________________________________________________
avg_pool (GlobalAveragePooling2D (None, 768)           0           block5_act[0][0]
____________________________________________________________________________________________________
drop_out (Dropout)               (None, 768)           0           avg_pool[0][0]
____________________________________________________________________________________________________
steering_angle (Dense)           (None, 1)             769         drop_out[0][0]
====================================================================================================
Total params: 521,665
Trainable params: 518,689
Non-trainable params: 2,976
____________________________________________________________________________________________________

    '''

    # x = Lambda(lambda input_img: input_img / 255.0 - 0.5, name="standardize_imgs")(intputs)

    x = Convolution2D(48, 5, 5, name="block1_conv")(inputs)
    x = BatchNormalization(name="block1_bn")(x)
    x = Activation('relu', name="block1_act")(x)

    x = Convolution2D(96, 5, 5, name="block2_conv")(x)
    x = BatchNormalization(name="block2_bn")(x)
    x = Activation('relu', name="block2_act")(x)
    x = MaxPooling2D(pool_size=(2, 2), name="block2_max_pool")(x)

    x = SeparableConvolution2D(192, 5, 5, name="block3_sepconv")(x)
    x = BatchNormalization(name="block3_bn")(x)
    x = Activation('relu', name="block3_act")(x)
    x = MaxPooling2D(pool_size=(2, 2), name="block3_max_pool")(x)

    x = SeparableConvolution2D(384, 3, 3, name="block4_sepconv")(x)
    x = BatchNormalization(name="block4_bn")(x)
    x = Activation('relu', name="block4_act")(x)

    x = SeparableConvolution2D(768, 3, 3, name="block5_sepconv")(x)
    x = BatchNormalization(name="block5_bn")(x)
    x = Activation('relu', name="block5_act")(x)

    x = GlobalAveragePooling2D(name="avg_pool")(x)

    x = Dropout(p=DROPOUT_P, name="drop_out")(x)

    outputs = Dense(1, name="steering_angle")(x)

    return outputs


if __name__ == '__main__':

    print("Gathering list of images from the center camera\n")
    img_files_center = [IMG_DIR + img_file for img_file in sorted(os.listdir(IMG_DIR))
                        if img_file.endswith('.jpg') and img_file.startswith('center')]

    print("Constructing input tensors (X_data and y_data)\n")
    X_data = []

    for img_file in img_files_center:
        img = mpimg.imread(img_file)
        X_data.append(preprocess_input(img))

    X_data = np.stack(X_data, axis=0)

    # gather associated steering angle
    df_y = pd.read_csv(DRIVE_LOG)
    y_data = df_y["steering"].values

    # stratified augmentation of training images based on steering angle bins (20 total bins)
    # in this case, each bin is augmented to 0.5 times the number of samples of the largest bin
    print("Augmenting input tensors\n")
    y_bins = np.linspace(-0.8, 0.8, 20)
    X_train_augs, y_train_augs = augment_images(X_data, y_data, multiplier=0.5, y_bins=y_bins)
    X_train_std, X_val_std, y_train_std, y_val_std = train_test_split(X_train_augs, y_train_augs, test_size=TEST_SIZE)

    print("Building nv_xception model\n")
    inputs = Input(shape=INPUT_SHAPE, name="input_layer")
    outputs = nv_xception(inputs)
    model = Model(input=inputs, output=outputs)
    model.compile(optimizer=RMSprop(lr=L_RATE), loss='mae', metrics=['mean_squared_error'])

    if LOAD_WEIGHTS:
        model.load_weights(MODEL_WEIGHT_NAME)

    with open(MODEL_NAME, 'w') as output_file:
        json_string = model.to_json()
        output_file.write(json_string)
        # json.dump(model.to_json(), output_file)

    model.summary()
    plot(model=model, to_file=MODEL_GRAPH_NAME, show_shapes=True)

    # train nv_xception model and save weights based on validation metric
    # each epoch costs around 300s on a NVIDIA 980Ti
    print("Training nv_xception model\n")
    cb_check_pt = callbacks.ModelCheckpoint(filepath=MODEL_WEIGHT_NAME, verbose=1, save_best_only=True)
    cb_reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0005)

    model.fit(X_train_std, y_train_std,
              validation_data=(X_val_std, y_val_std),
              nb_epoch=EPOCHS, batch_size=BATCH_SIZE,
              callbacks=[cb_check_pt, cb_reduce_lr])
