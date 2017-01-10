"""
Live trainer script for Udacity SDC sim
- Control car with Keras model
- Override with manual control
- Train model during manual control
- Record data from simulator

Usage guide
- accelerate/decelerate with "up/down" arrow keys
- navigation with "left/right" arrow keys
- change steering angle to "0" with "c" key
- toggle manual/auto mode with "x" key
- while in manual mode, toggle training with "z" key
- toggle image recording with "r" key

"""
# authored by Thomas Antony
# modified by Jingzhe Hu @ 01/07/2017
# changed preprocessing and added rudimentary data recording feature

training_batch_size = 48
checkpoint_filename = './checkpoint.h5'

# hdf_recordings = "./recorded_data/data.h5"
dir_recordings = './recorded_data/'
steering_angle_recordings = "./recorded_data/steering_angle.csv"
learning_rate = 0.001

## PLEASE DO NOT EDIT PAST THIS POINT
__author__ = 'Thomas Antony'

import os
import csv
import sys
import h5py
import time
import tkinter
import argparse
import base64
import json
import cv2

import numpy as np
import pandas as pd
from server import ControlServer
from platform import system as platform
from datetime import datetime
import matplotlib.image as mpimg

import socketio
import eventlet
import eventlet.wsgi
from flask import Flask
from functools import partial

from keras.models import model_from_json
from keras.optimizers import Adam, RMSprop


class LiveTrainer(object):
    def __init__(self, model):
        # Control variables
        self.steering_angle = 0
        self.throttle = 0

        # State
        self.speed = 0

        # Parameters
        self.speed_limit = 30
        self.turn_rate = 0.5
        self.steering_limit = 15. / 25.
        self.centering_torque = 0.01 / 25.

        # Helper functions
        self.turn_left = partial(self.turn, direction=-1)
        self.turn_right = partial(self.turn, direction=+1)
        self.speed_up = partial(self.speed_control, direction=+1)
        self.slow_down = partial(self.speed_control, direction=-1)

        # Control server for getting data from simulator
        self.control_srv = ControlServer()
        self.control_srv.register_callback(self)  # Callback for telemetry

        self.mode = 'auto'  # can be 'auto' or 'manual'
        self.is_training = False  # Trains model if set to true
        self.is_recording = False  # Record camera images and steering angle

        # self.recorded_images = []
        # self.recorded_steering_angles = []

        self.model = model
        self.current_X = []  # List of images
        self.current_Y = []  # List of steering angles

        # Performance metrics
        self.start_time = None
        self.last_switch_time = None
        self.auto_time = 0

    def init_gui(self):
        # Create the root window
        self.root = tkinter.Tk()
        self.root.geometry('350x100+490+550')
        self.root.title('SDC Live Trainer')

        # Create a label with status
        self.status = tkinter.StringVar()
        label = tkinter.Label(self.root, width=350, height=100,
                              textvariable=self.status)
        label.pack(fill=tkinter.BOTH, expand=1)

        # Bind key event handlers
        self.root.bind('<Left>', lambda e: self.turn_left())
        self.root.bind('<Right>', lambda e: self.turn_right())
        self.root.bind('<Up>', lambda e: self.speed_up())
        self.root.bind('<Down>', lambda e: self.slow_down())
        self.root.bind('<Key>', self.keydown)

        self.update_status()

        # Start UI loop
        eventlet.spawn_after(1, self.main_loop)

    def start_server(self):
        self.control_srv.start()  # Start server

    def focus_gui(self):
        self.root.focus_force()

        # OSX code for focusing window
        if platform() == 'Darwin':
            os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')

    def main_loop(self):
        self.focus_gui()
        while True:
            try:
                self.root.update_idletasks()
                self.root.update()
            except:
                pass
            eventlet.sleep(0.01)

    def update_status(self):
        str_mode = 'Autopilot Engaged' if self.mode == 'auto' else 'Manual override'
        str_train_text = 'Training neural net ...' if self.is_training else ''
        str_recording = 'Recording images and steering angle ...' if self.is_recording else ''

        if self.start_time is not None:
            now = time.time()
            total_time = now - self.start_time
            auto_time = self.auto_time
            if self.mode == 'auto':
                auto_time += (now - self.last_switch_time)

            str_rating = auto_time / total_time
        else:
            str_rating = 0.0
        status_txt = '{0}\nAutnomous rating: {1:.2%}\n{2}\n{3}\nSpeed = {4:4.2f} mph, Steering angle = {5:4.2f} deg'

        self.status.set(status_txt.format(str_mode, str_rating, str_train_text, str_recording, self.speed, self.steering_angle * 25))

    def update_timers(self):
        """
        Triggered after a mode change or at start.
        """
        # Update timers for autonomous mode
        if self.mode == 'auto':
            self.last_switch_time = time.time()
        else:
            self.auto_time += time.time() - self.last_switch_time

    def keydown(self, event):
        if event.char == 'q':
            self.root.destroy()
            os._exit(0)  # Sledgehammer

        elif event.char == 'c' or event.char == 'C':
            self.reset_steering()

        elif event.char == 'x' or event.char == 'X':
            if self.mode == 'manual':
                self.is_training = False  # No training in autonomous mode
                self.mode = 'auto'
            else:
                self.mode = 'manual'
            self.update_timers()

        elif event.char == 'z' or event.char == 'Z':
            # Toggle flag (only in manual mode)
            if self.mode == 'manual':
                self.is_training = not self.is_training

        elif event.char == "r" or event.char == "R":
            self.is_recording = not self.is_recording

    def speed_control(self, direction):
        """
        direction = +1 for increase, -1 for decrease
        """
        self.speed += direction * 1

        self.speed = max(0, self.speed)
        self.speed = min(self.speed_limit, self.speed)

        self.update_status()

    def update_throttle(self, data):
        """
        Implements P-controller for speed
        """
        throttle_max = 1.0
        throttle_min = -1.0

        K = 0.35  # Proportional gain

        self.throttle = (self.speed - data['speed']) * K
        self.throttle = min(throttle_max, self.throttle)
        self.throttle = max(throttle_min, self.throttle)

    def update_steering(self, data):
        """
        Implements a simple centering torque for the manual steering
        """
        if self.mode == 'manual':
            if abs(self.steering_angle) < self.centering_torque:
                self.steering_angle = 0.0
            elif self.steering_angle > 0:
                self.steering_angle -= self.centering_torque
            elif self.steering_angle < 0:
                self.steering_angle += self.centering_torque

    def turn(self, direction=None):
        """
        direction = +1 for right, -1 for left
        """
        self.steering_angle += direction * self.turn_rate / 25.
        self.steering_angle = max(self.steering_angle, -self.steering_limit)
        self.steering_angle = min(self.steering_angle, +self.steering_limit)

        self.update_status()

    def reset_steering(self):
        self.steering_angle = 0.0
        self.update_status()

    def preprocess_input(self, img):
        ''' Crop, resize and convert input image from RGB to HLS colorspace

        :param self:
        :param img: np array of uint8
        :return: preprocessed image
        '''

        img = cv2.resize(img[60:140, 40:280], (200, 66))

        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype("float32")

    def predict_steering(self, data):
        x = self.preprocess_input(data['image'])
        x = x[None, :, :, :]  # Extend dimension
        return float(model.predict(x, batch_size=1))

    def save_batch(self, data):
        """
        Saves training data in current batch to disk.
        """
        # TODO: Implement save_batch()
        pass

    def train_model(self, model, X_train, y_train):
        h = model.fit(X_train, y_train,
                      nb_epoch=1, verbose=0, batch_size=training_batch_size)
        model.save_weights(checkpoint_filename)
        print('loss : ', h.history['loss'][-1])
        return model

    def process_data(self, data):
        """
        If current batch is full, train the model, save data and reset cache.
        else just save data into batch
        """
        img = self.preprocess_input(data['image'])
        steering_angle = self.steering_angle

        # randomly flip training image horizontally
        flip = np.random.choice([True, False])
        if flip:
            img = cv2.flip(img, 1)
            steering_angle = -steering_angle

        self.current_X.append(img)
        self.current_Y.append(steering_angle)

        if len(self.current_Y) == training_batch_size:
            X_train = np.array(self.current_X)
            y_train = np.array(self.current_Y)

            print('Training model ...')
            self.train_model(self.model, X_train, y_train)

            self.save_batch((X_train, y_train))

            # Reset internal batch
            self.current_X = []
            self.current_Y = []

    # Callback functions triggered by ControlServer
    def handle_connect(self, sid):
        self.start_time = time.time()  # Reset timer
        self.auto_time = 0.0
        self.last_switch_time = None
        self.update_timers()

        # Focus window when simulator connects
        self.focus_gui()

    def handle_telemetry(self, data):

        if self.mode == 'auto':
            self.steering_angle = self.predict_steering(data)
        elif self.mode == 'manual':
            steering_angle = self.steering_angle

            if self.is_training:
                self.process_data(data)

        if self.is_recording:
            # Todo: write to hdf5 store instead
            img_filename = "center_"+str(datetime.now()) + '.jpg'
            mpimg.imsave(os.path.join(dir_recordings, img_filename),
                         data["image"].astype("uint8"))
            # print(img_filename)

            # with pd.HDFStore(hdf_recordings) as hdf_file:
            #     hdf_file.append()

            with open(steering_angle_recordings, "a") as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=",")

                if not os.path.exists(steering_angle_recordings):
                    csv_writer.writerow(['center', 'steering'])

                csv_writer.writerow([img_filename, self.steering_angle])

        # Send current control variables to simulator
        self.control_srv.send_control(self.steering_angle, self.throttle)

        # Update UI
        self.update_status()

        # Steering dynamics and speed controller
        self.update_steering(data)
        self.update_throttle(data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
                        help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(jfile.read())

    rmsprop = RMSprop(lr=learning_rate)
    model.compile(rmsprop, "mae")
    weights_file = args.model.replace('json', 'h5')

    if os.path.exists(weights_file):
        model.load_weights(weights_file)

    if not os.path.exists(dir_recordings):
        os.makedirs(dir_recordings)

    driver = LiveTrainer(model)
    driver.init_gui()
    driver.start_server()
