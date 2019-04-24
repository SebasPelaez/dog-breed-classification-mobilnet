import base64
import io
import numpy as np
import os

from PIL import Image

from flask import Flask, render_template,request, jsonify

import keras
import keras.models
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils

import load
import utils


#initalize our flask app
app = Flask(__name__)

params = utils.yaml_to_dict('config.yml')

global model, graph
model, graph = load.initialize_variables(params)
  
def prepare_image(image, target):
  # if the image mode is not RGB, convert it
  if image.mode != "RGB":
    image = image.convert("RGB")

  # resize the input image and preprocess it
  image = image.resize(target)
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  image = imagenet_utils.preprocess_input(image)

  # return the processed image
  return image

@app.route('/')
def index():
  return render_template("index.html")

@app.route("/predict/", methods=["POST"])
def predict():

  message = request.get_json(force=True)
  encoded = message['image']
  decoded = base64.b64decode(encoded)

  image = Image.open(io.BytesIO(decoded))
  processed_image = prepare_image(image, target=(224, 224))

  with graph.as_default():
    prediction = model.predict(x=processed_image,batch_size=params['batch_size'], verbose=1)
    print(prediction)
    print(np.argmax(prediction,axis=1))
    print("debug3")
  
    response = {
      'prediction': {
        'pred': prediction
      }
    }

    return jsonify(response)

if __name__ == "__main__":
  #decide what port to run the app in
  port = int(os.environ.get('PORT', 5000))
  #run the app locally on the givn port
  app.run(host='0.0.0.0', port=port, debug=True)
  #optional if we want to run in debugging mode
  #app.run(debug=True)