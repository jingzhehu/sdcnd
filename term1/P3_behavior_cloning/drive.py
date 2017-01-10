# authored by Thomas Antony
# modified by Jingzhe Hu @ 01/06/2017
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import cv2

from keras.models import model_from_json
from keras.preprocessing.image import img_to_array

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

tf.python.control_flow_ops = control_flow_ops

sio = socketio.Server()
app = Flask(__name__)
model = None


def preprocess_input(img):
    ''' Crop, resize and convert input image from RGB to HLS colorspace

    :param img: np array of uint8
    :return: preprocessed image in float32
    '''

    img = cv2.resize(img[60:140, 40:280], (200, 66))

    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype("float32")/255.0 - 0.5


@sio.on('telemetry')
def telemetry(sid, data):
    # current steering angle of the car
    steering_angle = data["steering_angle"]

    # current throttle of the car
    throttle = data["throttle"]

    # current speed of the car
    speed = float(data["speed"])

    # current image from the center camera of the car
    img_string = data["image"]
    img = Image.open(BytesIO(base64.b64decode(img_string)))

    # preprocess image from center camera
    img = np.array(img, dtype=np.uint8)
    img = preprocess_input(img)
    img = img[None, :, :, :]

    # predict steering angle from preprocessed image
    # model accepts (66, 200, 3, dtype=float32) as input
    steering_angle = float(model.predict(img, batch_size=1))

    throttle_max = 1.0
    throttle_min = -1.0
    steering_threshold = 3. / 25

    # targets for speed controller
    nominal_set_speed = 30
    steering_set_speed = 30

    K = 0.35  # proportional gain

    # slow down for turns
    if abs(steering_angle) > steering_threshold:
        set_speed = steering_set_speed
    else:
        set_speed = nominal_set_speed

    throttle = (set_speed - speed) * K
    throttle = min(throttle_max, throttle)
    throttle = max(throttle_min, throttle)

    # else don't change from previous
    print("steering angle {:6.3f}, throttle {:6.3f}".format(steering_angle, throttle))
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer",
             data={'steering_angle': steering_angle.__str__(),
                   'throttle': throttle.__str__()}
             , skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
                        help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(jfile.read())

    model.compile("rmsprop", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
