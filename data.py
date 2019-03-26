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

def input_fn(sources, train, params):

  def parse_image(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    image_decoded.set_shape((None, None, 3))
    return image_decoded, label

  def resize_image(image, label):

    resized_images = tf.image.resize_images( 
        images = image,
        size = [224, 224],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )

    return resized_images, label


  image_list, label_list = zip(*sources)

  image_list = list(image_list)
  label_list = list(label_list)

  image_data_set = tf.data.Dataset.from_tensor_slices(image_list)
  label_data_set = tf.data.Dataset.from_tensor_slices(label_list)

  data_set = tf.data.Dataset.zip((image_data_set,label_data_set))

  data_set = data_set.map(parse_image, num_parallel_calls=4)
  data_set = data_set.map(resize_image, num_parallel_calls=4)

  if train:
    data_set = data_set.shuffle(buffer_size=params['shuffle_buffer'])
    data_set = data_set.repeat()

  data_set = data_set.batch(params['batch_size'])
  iterator = data_set.make_one_shot_iterator()

  images_batch, labels_batch = iterator.get_next()

  features = {'image': images_batch}
  y = labels_batch
    
  return features, y