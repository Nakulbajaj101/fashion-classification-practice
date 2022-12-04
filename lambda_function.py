
from io import BytesIO
from urllib import request

import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image


CLASSES = ['dress',
 'hat',
 'longsleeve',
 'outwear',
 'pants',
 'shirt',
 'shoes',
 'shorts',
 'skirt',
 't-shirt']


def download_image(url):
    """Function to download the image"""
    
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def preprocess_input(x):
    """Function to preproces image for Tensorflow lite"""
    
    x /= 127.5
    x -= 1.0
    return x


class Preprocessor():
    """Image preprocessing class"""

    def __init__(self, url, size=299):
        self.url = url
        self.size = size

    def preprocess_image(self):
        image = download_image(self.url)
        image = image.resize(size=(self.size, self.size), resample=Image.Resampling.NEAREST)
        img_arr = np.array(image, dtype='float32')
        X = np.array([img_arr])
        X = preprocess_input(X)
        return X


def load_model(model_path : str="./model.tflite"):
    """Function to load the tflite model and return the object and the indexes"""
    
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    return interpreter, input_index, output_index



def predict(url, interpreter, input_index, output_index):
    """Function to predict the model"""
    
    preprocessor  = Preprocessor(url=url)
    X = preprocessor.preprocess_image()
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)
    predictions = pred[0].tolist()

    pred_classes = dict(zip(CLASSES, predictions))
    prediction = {k: v for k, v in sorted(pred_classes.items(), key=lambda item: item[1], reverse=True)}

    return prediction


interpreter, input_index, output_index = load_model()


def lambda_handler(event, context):
    """Lambda handler"""

    url = event['url']
    prediction = predict(url=url, interpreter=interpreter, input_index=input_index, output_index=output_index)
    return prediction
