import cv2
import os
import sys

import tensorflow as tf
import numpy as np

from tensorflow import keras

import model


def initialize_variables(params):

  base_model = tf.keras.applications.MobileNet(input_shape=params['image_shape'],include_top=False,weights='imagenet')
  base_model.trainable = False

  mobilnet_tiny = model.MobilNet_Architecture_Tiny(
    width_multiplier=params['width_multiplier'],
    depth_multiplier=params['depth_multiplier'],
    num_classes=params['num_classes'],
    dropout_rate=params['dropout_rate'],
    regularization_rate=params['regularization_rate'])

  loaded_model = tf.keras.Sequential([
    base_model,
    mobilnet_tiny])

  optimizer = tf.keras.optimizers.Adam(lr=params['learning_rate'])
  loaded_model.compile(optimizer=optimizer,loss=params['loss'],metrics=['accuracy'])
  loaded_model.load_weights(os.path.join(params['model_dir'], 'tf_ckpt'))
  #loaded_model._make_predict_function()

  graph = tf.get_default_graph()
  
  return loaded_model, graph