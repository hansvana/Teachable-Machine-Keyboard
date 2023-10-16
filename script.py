from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import pydirectinput as pdi
import time
from collections import deque
import os

# Check if keras_Model.h5 exists
if not (os.path.isfile("keras_Model.h5") and os.path.isfile("labels.txt")):
    print("Error: 'keras_Model.h5' and/or 'labels.txt' does not exist in the current directory.")
    exit(1)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

pdi.PAUSE = 0

def crop_square(img, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h,w])

    # Centralize and crop
    crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)

    return resized

global prediction_history
prediction_history = deque(maxlen=5)
global last_key_pressed
last_key_pressed = ''

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()    

    if ret:

        # Resize the raw image into (224-height,224-width) pixels
        # image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = crop_square(image, 224)

        image = cv2.flip(image, 1)

        # Show the image in a window
        cv2.imshow("Webcam Image", image)

        # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Predicts the model
        prediction = model.predict(image, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        current_prediction = class_name[2:].strip()

        # adds current prediction to the history
        prediction_history.append(current_prediction)

        # Check if all 5 items in the deque are the same
        if len(set(prediction_history)) == 1:
            if last_key_pressed != current_prediction:
                if current_prediction != 'neutral':
                    pdi.press(current_prediction)            
                print(current_prediction)
                last_key_pressed = current_prediction

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break



camera.release()
cv2.destroyAllWindows()
