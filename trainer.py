import tensorflow as tf

import argparse

import data
import model
import utils

tf.logging.set_verbosity(tf.logging.INFO)

def train_model(params):

  estimator = tf.estimator.Estimator(
    model.model_fn,
    model_dir=params['model_dir'],
    params=params,
    config=tf.estimator.RunConfig(
      save_checkpoints_steps=params['save_checkpoints_steps'],
      save_summary_steps=params['save_summary_steps'],
      log_step_count_steps=params['log_frequency'],
      keep_checkpoint_max=params['keep_checkpoint_max']
    )
  )

  train_sources = data._sources(params)
  test_sources = data._sources(params,mode='validation')

  train_spec = tf.estimator.TrainSpec(
    lambda: data.input_fn(train_sources, True, params),
    max_steps=params['max_steps']
  )

  eval_spec = tf.estimator.EvalSpec(
    lambda: data.input_fn(test_sources, False, params),
    steps=params['eval_steps'],
    start_delay_secs=params['start_delay_secs']
  )

  tf.logging.info("Start experiment....")

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

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