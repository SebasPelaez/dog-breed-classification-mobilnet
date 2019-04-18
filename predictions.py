import cv2
import os
import sys

import tensorflow as tf
import numpy as np

from tensorflow import keras

import model

def data_generator(params):
  
  images_to_predict_path = os.path.join(params['data_dir'],'images_to_predict')

  datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
  generator = datagen.flow_from_directory(
    images_to_predict_path, 
    target_size=(224, 224), 
    class_mode=None, 
    batch_size=params['batch_size'])

  return generator

def make_predictions(data_to_predict, params):
  
  base_model = tf.keras.applications.MobileNet(input_shape=params['image_shape'],include_top=False,weights='imagenet')
  base_model.trainable = False

  mobilnet_tiny = model.MobilNet_Architecture_Tiny(
    width_multiplier=params['width_multiplier'],
    depth_multiplier=params['depth_multiplier'],
    num_classes=params['num_classes'],
    dropout_rate=params['dropout_rate'],
    regularization_rate=params['regularization_rate'])

  net = tf.keras.Sequential([
    base_model,
    mobilnet_tiny])

  optimizer = tf.keras.optimizers.Adam(lr=params['learning_rate'])
  net.compile(optimizer=optimizer,loss=params['loss'],metrics=['accuracy'])
  net.load_weights(os.path.join(params['model_dir'], 'tf_ckpt'))
  predictions = net.predict(x=data_to_predict, batch_size=params['batch_size'], verbose=1)

  return np.argmax(predictions,axis=1)