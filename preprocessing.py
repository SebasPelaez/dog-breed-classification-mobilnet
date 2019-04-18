import json
import os
import scipy.io
import tarfile
import wget

import numpy as np
import pandas as pd

import utils

def download_data(params):
  
  url_file = params['url_dataset']
  url_meta = params['url_list']

  if not os.path.exists(params['data_dir']):
    os.makedirs(params['data_dir'])
    os.makedirs(os.path.join(params['data_dir'],params['images_to_predict']))

  wget.download(url_file, params['data_dir'])
  wget.download(url_meta, params['data_dir'])

def extract_data(params):

  tar_file = os.path.join(params['data_dir'],params['compressed_data_name'])
  tar_meta = os.path.join(params['data_dir'],params['compressed_list_name'])

  _decompress(file_to_decompress=tar_file,location_to_decompress=params['data_dir'])
  _decompress(file_to_decompress=tar_meta,location_to_decompress=os.path.join(params['data_dir'],params['data_dir_list']))

def _decompress(file_to_decompress, location_to_decompress):

  tar = tarfile.open(file_to_decompress, 'r:*')
  for item in tar:
    tar.extract(item,location_to_decompress)

def make_id_label_map(params):

  labels_json_path = os.path.join(params['data_dir'],params['label_id_json'])

  if not os.path.isfile(labels_json_path):
    id_label_map = dict()
    images_path = os.path.join(params['data_dir'],params['data_dir_images'])
    for numeric_label,string_label in enumerate(os.listdir(images_path)):
      index_of = string_label.index('-')
      label = string_label[index_of+1:]
      id_label_map[label] = numeric_label
    
    with open(labels_json_path,'w') as file:
      json.dump(id_label_map,file)

def split_data(params):

  train_dataset_mat_path = os.path.join(params['data_dir'],params['data_dir_list'],params['train_mat_file'])
  train_dataset_mat = scipy.io.loadmat(train_dataset_mat_path)
  train_dataset_mat['labels'] = train_dataset_mat['labels'] - 1 

  test_dataset_mat_path = os.path.join(params['data_dir'],params['data_dir_list'],params['test_mat_file'])
  test_dataset_mat = scipy.io.loadmat(test_dataset_mat_path)
  test_dataset_mat['labels'] = test_dataset_mat['labels'] - 1

  training_dev_df = _make_data_frame(file_mat=train_dataset_mat, shuffle=True, params=params)
  test_df = _make_data_frame(file_mat=test_dataset_mat, shuffle=True, params=params)

  if params['num_classes'] < 120:
    classes = np.arange(params['num_classes'])

    training_dev_filter = training_dev_df['labels'].isin(classes)
    training_dev_df = training_dev_df[training_dev_filter]

    test_filter = test_df['labels'].isin(classes)
    test_df = test_df[test_filter]

  training_df = pd.DataFrame()
  validation_df = pd.DataFrame()
  
  for label in np.arange(params['num_classes']):

    df_group = training_dev_df[training_dev_df['labels'] == label]
    
    training_sample = df_group.sample(frac=0.8)
    validation_sample = df_group.drop(training_sample.index)

    training_df = training_df.append(training_sample)
    validation_df = validation_df.append(validation_sample)

  training_df.to_csv(os.path.join(params['data_dir'],params['training_data']), header=True, index=None, sep='\t')
  validation_df.to_csv(os.path.join(params['data_dir'],params['validation_data']), header=True, index=None, sep='\t')
  
  test_df.to_csv(os.path.join(params['data_dir'],params['test_data']), header=True, index=None, sep='\t')

def _make_lists(file_mat, params):
    
  images_list = list()
  label_list = list()

  for images,labels in zip(file_mat['file_list'][:],file_mat['labels']):
    images_list.append(os.path.join(params['data_dir'],params['data_dir_images'],images[0][0]))
    label_list.append(labels[0])
      
  return images_list,label_list

def _make_data_frame(file_mat, shuffle, params):

  data_in_list = _make_lists(file_mat, params)
  data_dict = {'images': data_in_list[0], 'labels': data_in_list[1]}
  df = pd.DataFrame.from_dict(data_dict)

  if shuffle:
    df = df.sample(frac=1).reset_index(drop=True)
  
  return df

if __name__ == '__main__':

  params = utils.yaml_to_dict('config.yml')
  download_data(params)
  extract_data(params)
  make_id_label_map(params)
  split_data(params)