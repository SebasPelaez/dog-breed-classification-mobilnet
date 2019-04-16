import os

import pandas as pd
import tensorflow as tf

def _sources(params, mode='training'):

  file_path = mode + '.txt'
  data_path = os.path.join(params['data_dir'],file_path)
  df = pd.read_csv(data_path, sep="\t", header=None)

  if params['shuffle']:
    df = df.sample(frac=1).reset_index(drop=True)
    
  def helper(img_path,lbl):
    img_fpath = os.path.join(params['data_dir_images'], img_path)    
    return img_fpath, lbl

  data = [helper(img_path,lbl) for img_path,lbl in zip(df[0],df[1])]
        
  return data

def flip_left_right(x):
  x = tf.image.random_flip_left_right(x)
  return x

def flip_up_down(x):
  x = tf.image.random_flip_up_down(x)
  return x

def rotate(x):
  return tf.image.rot90(x, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

def color_saturation(x):
  return tf.image.random_saturation(x, 0.6, 1.6)

def color_brightness(x):
  return tf.image.random_brightness(x, 0.05)

def color_contrast(x):
  return tf.image.random_contrast(x, 0.7, 1.3)

def input_fn(sources, train, params):

  def load_and_preprocess_images(row):

    image_string = tf.read_file(row['image'])
    image_decoded = tf.image.decode_image(image_string)
    image_decoded.set_shape((None, None, 3))
    img = tf.image.resize_images( 
      images = image_decoded,
      size = [224, 224],
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    img = tf.cast(img, tf.float32) / 255

    return img, row['label']

  images, labels = zip(*sources)

  data_set = tf.data.Dataset.from_tensor_slices({
    'image': list(images),
    'label': list(labels)})

  # Add augmentations
  augmentations = [flip_left_right, flip_up_down, rotate, color_brightness, color_contrast, color_saturation]

  data_set = data_set.map(load_and_preprocess_images, num_parallel_calls=4)

  for f in augmentations:
    data_set = data_set.map(lambda x,y: (tf.cond(tf.random_uniform([], 0, 1) > 0.75, lambda: f(x), lambda: x),y), num_parallel_calls=4)

  if train:
    data_set = data_set.shuffle(buffer_size=params['batch_size']*4)
    data_set = data_set.repeat(count=params['num_epochs'])

  data_set = data_set.batch(params['batch_size'])
  data_set = data_set.prefetch(1)
    
  return data_set