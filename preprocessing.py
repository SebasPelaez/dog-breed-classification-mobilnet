import json
import os
import scipy.io
import tarfile
import wget

import pandas as pd

import utils

def download_data(params):
  
  url_file = params['url_dataset']
  url_meta = params['url_list']

  if not os.path.exists(params['data_dir']):
    os.makedirs(params['data_dir'])

  wget.download(url_file, params['data_dir'])
  wget.download(url_meta, params['data_dir'])

def extract_data(params):

  tar_file = os.path.join(params['data_dir'],params['compressed_data_name'])
  tar_meta = os.path.join(params['data_dir'],params['compressed_list_name'])

  _decompress(file_to_decompress=tar_file,location_to_decompress=params['data_dir'])
  _decompress(file_to_decompress=tar_meta,location_to_decompress=params['data_dir_list'])

def _decompress(file_to_decompress, location_to_decompress):

  tar = tarfile.open(file_to_decompress, 'r:*')
  for item in tar:
    tar.extract(item,location_to_decompress)

def make_id_label_map(params):

  labels_json_path = os.path.join(params['data_dir'],params['label_id_json'])

  if not os.path.isfile(labels_json_path):
    id_label_map = dict()
    for numeric_label,string_label in enumerate(os.listdir(params['data_dir_images'])):
      index_of = string_label.index('-')
      label = string_label[index_of+1:]
      id_label_map[label] = numeric_label+1
    
    with open(labels_json_path,'w') as file:
      json.dump(id_label_map,file)

def split_data(params):

  train_dataset_mat_path = os.path.join(params['data_dir_list'],params['train_mat_file'])
  train_dataset_mat = scipy.io.loadmat(train_dataset_mat_path)

  test_dataset_mat_path = os.path.join(params['data_dir_list'],params['test_mat_file'])
  test_dataset_mat = scipy.io.loadmat(test_dataset_mat_path)

  training_dev_df = _make_data_frame(file_mat=train_dataset_mat)

  training_df = training_dev_df.sample(frac = 0.85)
  validation_df = training_dev_df.drop(training_df.index)

  training_df.to_csv(os.path.join(params['data_dir'],params['training_data']), header=None, index=None, sep='\t')
  validation_df.to_csv(os.path.join(params['data_dir'],params['validation_data']), header=None, index=None, sep='\t')

  test_df = _make_data_frame(file_mat=test_dataset_mat)
  test_df.to_csv(os.path.join(params['data_dir'],params['test_data']), header=None, index=None, sep='\t')

def _make_lists(file_mat):
    
  images_list = list()
  label_list = list()

  for images,labels in zip(file_mat['file_list'][:],file_mat['labels']):
    images_list.append(images[0][0])
    label_list.append(labels[0])
      
  return images_list,label_list

def _make_data_frame(file_mat):

  data_in_list = _make_lists(file_mat)
  data_dict = {'images': data_in_list[0], 'labels': data_in_list[1]}
  df = pd.DataFrame.from_dict(data_dict)
  df = df.sample(frac=1).reset_index(drop=True)
  
  return df


if __name__ == '__main__':

  params = utils.yaml_to_dict('config.yml')
  download_data(params)
  extract_data(params)
  make_id_label_map(params)
  split_data(params)