# import the necessary packages
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np

from config import num_classes
from data_generator import random_choice, safe_crop, to_bgr
from model import build_model
from utils import get_best_model
import hug
from io import BytesIO

model = build_model()
model.load_weights(get_best_model())

@hug.get()
def get_status():
    print(model.summary())


img_rows, img_cols = 320, 320

@hug.output_format.on_valid('img/png')
@hug.post("/predict")
def classify_image(image_data):    
    img_array = np.fromstring(image_data, dtype=np.uint8)
    image = cv.imdecode(img_array, cv.IMREAD_COLOR)
    image_size = image.shape[:2]

    x, y = random_choice(image_size)
    image = safe_crop(image, x, y)
    print('Start processing image...')

    x_test = np.empty((1, img_rows, img_cols, 3), dtype=np.float32)
    x_test[0, :, :, 0:3] = image / 255.

    out = model.predict(x_test)
    out = np.reshape(out, (img_rows, img_cols, num_classes))
    out = np.argmax(out, axis=2)
    out = to_bgr(out)

    ret = image * 0.6 + out * 0.4
    ret = ret.astype(np.uint8)

    K.clear_session()
    is_success, buffer = cv.imencode(".png", out)
    buffer
    out_bytes = BytesIO(buffer)
    return out_bytes