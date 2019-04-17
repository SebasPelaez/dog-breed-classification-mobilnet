import os

import pandas as pd
import tensorflow as tf

import utils

from tensorflow import keras

def data_generator(params, mode):

  id_label_map = utils.load_id_label_map(params)

  file_path = mode + '.txt'
  data_path = os.path.join(params['data_dir'],file_path)
  df = pd.read_csv(data_path, sep="\t", header=0)

  if params['shuffle']:
    df = df.sample(frac=1).reset_index(drop=True)

  datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

  generator = datagen.flow_from_dataframe(
    dataframe = df,
    x_col='images',
    y_col='labels',
    target_size=(224, 224),
    batch_size=params['batch_size'],
    class_mode=params['class_mode'])

  return generator