import json
import os
import yaml

def yaml_to_dict(yml_path):
  with open(yml_path, 'r') as stream:

    try:
      params = yaml.load(stream)
    except yaml.YAMLError as exc:
      print(exc)

  return params

def load_id_label_map(params):
  label_json_path = os.path.join(params['data_dir'], params['label_id_json'])
  with open(label_json_path, 'r') as file:
    label_dict = json.load(file) 
  
  labels_json_dict = {k:v for k,v in enumerate(label_dict)}

  return labels_json_dict