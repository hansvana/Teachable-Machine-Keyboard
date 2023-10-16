# Teachable-Machine-Keyboard
A python script to use a Tensorflow Keras model (e.g. trained with Teachable Machine) to control your keyboard

## Prepare

1. Install the latest version of Python that is supported by Tensorflow (See the [Software Requirements](https://www.tensorflow.org/install/pip) for Tensorflow). This script was tested with Python 3.11. Make sure to 'Add Python to the PATH' during the installation.
2. Open a console, type `python --version` to see if Python is installed correctly, it should return the version number.
3. To install all the scripts' dependencies, type `pip install tensorflow opencv-python pydirectinput`.
4. Download and extract script.py and RUN.bat into a folder on your pc.

## Run (short version)

1. Train an Teachable Machine image model with 5 classes named exactly `neutral` `up` `down` `left` `right`
2. Download as a Tensorflow Keras model and place the files in the folder you created earlier
3. Make sure the webcam is not used by any other application like the browser
4. Run RUN.bat

## Run (long version)

1. See the included pdf file for instructions aimed at students
