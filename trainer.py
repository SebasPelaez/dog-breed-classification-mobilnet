import argparse
import json
import keras
import os

import tensorflow as tf

import data
import model
import utils

tf.logging.set_verbosity(tf.logging.INFO)

def train_model(params):

  train_generator = data.data_generator(params, mode='training')
  val_generator = data.data_generator(params,mode='validation')

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

  cp_callback = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(params['model_dir'], 'tf_ckpt'), 
    save_weights_only=True, 
    verbose=1,
    period=5)

  tb_callback = tf.keras.callbacks.TensorBoard(
    os.path.join(params['model_dir'], 'logs'))

  optimizer = tf.keras.optimizers.Adam(lr=params['learning_rate'])

  net.compile(optimizer=optimizer,loss=params['loss'],metrics=[f1_score,'accuracy'])

  steps_per_epoch = train_generator.n // params['batch_size']
  validation_steps = val_generator.n // params['batch_size']

  history = net.fit_generator(
    train_generator,
    steps_per_epoch = steps_per_epoch,
    epochs=params['num_epochs'], 
    workers=4,
    validation_data=val_generator, 
    validation_steps=validation_steps,
    callbacks=[cp_callback, tb_callback])

  # Save it under the form of a json file
  with open(os.path.join(params['model_dir'], 'history.json'),'w') as file:
    json.dump(history.history,file)


def f1_score(y_true, y_pred):
  #https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
  #https://machinelearningmastery.com/check-point-deep-learning-models-keras/
  def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + keras.backend.epsilon())
    return recall

  def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + keras.backend.epsilon())
    return precision

  precision = precision(y_true, y_pred)
  recall = recall(y_true, y_pred)
  return 2*((precision*recall)/(precision+recall+keras.backend.epsilon()))

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', help="path to configuration file", default='config.yml')
  parser.add_argument('-v', '--verbosity', default='INFO',
    choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARM'],
  )
  
  args = parser.parse_args()
  tf.logging.set_verbosity(args.verbosity)

  params = utils.yaml_to_dict(args.config)
  tf.logging.info("Using parameters: {}".format(params))
  train_model(params)