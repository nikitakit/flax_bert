# Copyright 2020 The FlaxBERT Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run sequence-level classification (and regression) fine-tuning."""

import datetime
import functools
import os
import typing

from absl import app
from absl import flags
import data
import dataclasses
from flax import nn
from flax import optim
import import_weights
import modeling
import training
# from flax.metrics import tensorboard
# from flax.training import checkpoints
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow.io import gfile

import datasets
import transformers
from transformers import BertTokenizerFast

import ml_collections
from ml_collections.config_flags import config_flags


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'output_dir', None,
    'The output directory where the model checkpoints will be written.')

config_flags.DEFINE_config_file(
  'config', None,
  'Hyperparameter configuration')


def get_config():
  config = FLAGS.config
  hf_config = transformers.AutoConfig.from_pretrained(config.init_checkpoint)
  assert hf_config.model_type == 'bert', 'Only BERT is supported.'
  model_config = ml_collections.ConfigDict({
    'vocab_size': hf_config.vocab_size,
    'hidden_size': hf_config.hidden_size,
    'num_hidden_layers': hf_config.num_hidden_layers,
    'num_attention_heads': hf_config.num_attention_heads,
    'hidden_act':  hf_config.hidden_act,
    'intermediate_size': hf_config.intermediate_size,
    'hidden_dropout_prob': hf_config.hidden_dropout_prob,
    'attention_probs_dropout_prob': hf_config.attention_probs_dropout_prob,
    'max_position_embeddings': hf_config.max_position_embeddings,
    'type_vocab_size': hf_config.type_vocab_size,
    # 'initializer_range': hf_config.initializer_range,
    # 'layer_norm_eps': hf_config.layer_norm_eps,
  })
  config.model = model_config
  return config


def get_output_dir(config):
  """Get output directory location."""
  output_dir = FLAGS.output_dir
  if output_dir is None:
    dataset_name = config.dataset_name.replace('/', '_')
    output_name = '{dataset_name}_{timestamp}'.format(
        dataset_name=dataset_name,
        timestamp=datetime.datetime.now().strftime('%Y%m%d_%H%M'),
    )
    output_dir = os.path.join('~', 'efficient_transformers', output_name)
    output_dir = os.path.expanduser(output_dir)
    print()
    print('No --output_dir specified')
    print('Using default output_dir:', output_dir, flush=True)
  return output_dir


def create_model(config, num_classes=2):
  """Create a model, starting with a pre-trained checkpoint."""
  model_kwargs = dict(
      config=config.model,
      n_classes=num_classes,
  )
  model_def = modeling.BertForSequenceClassification.partial(**model_kwargs)
  if config.init_checkpoint:
    initial_params = import_weights.load_params(
        init_checkpoint=config.init_checkpoint,
        hidden_size=config.model.hidden_size,
        num_attention_heads=config.model.num_attention_heads,
        num_classes=num_classes)
  else:
    with nn.stochastic(jax.random.PRNGKey(0)):
      _, initial_params = model_def.init_by_shape(
          jax.random.PRNGKey(0),
          [((1, config.max_seq_length), jnp.int32),
           ((1, config.max_seq_length), jnp.int32),
           ((1, config.max_seq_length), jnp.int32),
           ((1, 1), jnp.int32)],
          deterministic=True)
  model = nn.Model(model_def, initial_params)
  return model


def create_optimizer(config, model):
  common_kwargs = dict(
    learning_rate=config.learning_rate,
    beta1=0.9,
    beta2=0.999,
    eps=1e-6,
  )
  optimizer_decay_def = optim.Adam(
    weight_decay=0.01, **common_kwargs)
  optimizer_no_decay_def = optim.Adam(
    weight_decay=0.0, **common_kwargs)
  decay = optim.ModelParamTraversal(lambda path, _: 'bias' not in path)
  no_decay = optim.ModelParamTraversal(lambda path, _: 'bias' in path)
  optimizer_def = optim.MultiOptimizer(
    (decay, optimizer_decay_def), (no_decay, optimizer_no_decay_def))
  optimizer = optimizer_def.create(model)
  return optimizer


def compute_loss_and_metrics(model, batch, rng):
  """Compute cross-entropy loss for classification tasks."""
  with nn.stochastic(rng):
    metrics = model(
        batch['input_ids'],
        (batch['input_ids'] > 0).astype(np.int32),
        batch['token_type_ids'],
        batch['label'])
  return metrics['loss'], metrics


def compute_classification_stats(model, batch):
  with nn.stochastic(jax.random.PRNGKey(0)):
    y = model(
        batch['input_ids'],
        (batch['input_ids'] > 0).astype(np.int32),
        batch['token_type_ids'],
        deterministic=True)
  return {
      'idx': batch['idx'],
      'label': batch['label'],
      'prediction': y.argmax(-1)
  }


def compute_regression_stats(model, batch):
  with nn.stochastic(jax.random.PRNGKey(0)):
    y = model(
        batch['input_ids'],
        (batch['input_ids'] > 0).astype(np.int32),
        batch['token_type_ids'],
        deterministic=True)
  return {
      'idx': batch['idx'],
      'label': batch['label'],
      'prediction': y[..., 0],
  }


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  config = get_config()

  datasets.logging.set_verbosity_error()
  dataset = datasets.load_dataset(config.dataset_path, config.dataset_name)
  os.environ['TOKENIZERS_PARALLELISM'] = 'true'
  tokenizer = BertTokenizerFast.from_pretrained(config.tokenizer)
  tokenizer.model_max_length = config.max_seq_length
  data_pipeline = data.ClassificationDataPipeline(dataset, tokenizer)

  num_train_examples = len(dataset['train'])
  num_train_steps = int(
      num_train_examples * config.num_train_epochs // config.train_batch_size)
  warmup_steps = int(config.warmup_proportion * num_train_steps)
  cooldown_steps = num_train_steps - warmup_steps

  is_regression_task = (
    dataset['train'].features['label'].dtype == 'float32')
  if is_regression_task:
    num_classes = 1
    compute_stats = compute_regression_stats
  else:
    num_classes = dataset['train'].features['label'].num_classes
    compute_stats = compute_classification_stats

  model = create_model(config, num_classes=num_classes)
  optimizer = create_optimizer(config, model)
  optimizer = optimizer.replicate()
  del model  # don't keep a copy of the initial model
  optimizer = training.harmonize_across_hosts(optimizer)

  learning_rate_fn = training.create_learning_rate_scheduler(
      factors='constant * linear_warmup * linear_decay',
      base_learning_rate=config.learning_rate,
      warmup_steps=warmup_steps,
      steps_per_cycle=cooldown_steps,
  )

  output_dir = get_output_dir(config)
  gfile.makedirs(output_dir)

  train_history = training.TrainStateHistory(learning_rate_fn)
  train_state = train_history.initial_state()

  if config.do_train:
    train_step_fn = training.create_train_step(
      compute_loss_and_metrics, clip_grad_norm=1.0)
    train_iter = data_pipeline.get_inputs(
      split='train', batch_size=config.train_batch_size, training=True)

    for step, batch in zip(range(0, num_train_steps), train_iter):
      optimizer, train_state = train_step_fn(optimizer, batch, train_state)

  if config.do_eval:
    eval_step = training.create_eval_fn(compute_stats)
    eval_results = []

    if config.dataset_path == 'glue' and config.dataset_name == 'mnli':
      validation_splits = ['validation_matched', 'validation_mismatched']
    else:
      validation_splits = ['validation']

    for split in validation_splits:
      eval_iter = data_pipeline.get_inputs(
          split=split, batch_size=config.eval_batch_size, training=False)
      eval_stats = eval_step(optimizer, eval_iter)
      eval_metric = datasets.load_metric(config.dataset_path, config.dataset_name)
      eval_metric.add_batch(
        predictions=eval_stats['prediction'],
        references=eval_stats['label'])
      eval_metrics = eval_metric.compute()
      prefix = 'eval_mismatched' if split == 'validation_mismatched' else 'eval'
      for name, val in sorted(eval_metrics.items()):
        line = f'{prefix}_{name} = {val:.06f}'
        print(line, flush=True)
        eval_results.append(line)

    eval_results_path = os.path.join(output_dir, 'eval_results.txt')
    with gfile.GFile(eval_results_path, 'w') as f:
      for line in eval_results:
        f.write(line + '\n')

  if config.do_predict:
    predict_step = training.create_eval_fn(compute_stats)
    predict_results = []

    path_map = {
        ('glue', 'cola', 'test'): 'CoLA.tsv',
        ('glue', 'mrpc', 'test'): 'MRPC.tsv',
        ('glue', 'qqp', 'test'): 'QQP.tsv',
        ('glue', 'sst2', 'test'): 'SST-2.tsv',
        ('glue', 'stsb', 'test'): 'STS-B.tsv',
        ('glue', 'mnli', 'test_matched'): 'MNLI-m.tsv',
        ('glue', 'mnli', 'test_mismatched'): 'MNLI-mm.tsv',
        ('glue', 'qnli', 'test'): 'QNLI.tsv',
        ('glue', 'rte', 'test'): 'RTE.tsv',
        # No eval on WNLI for now. BERT accuracy on WNLI is below baseline,
        # unless a special training recipe is used.
        # ('glue/wnli', 'test'): 'WNLI.tsv',
    }
    label_sets = {
        ('glue','cola'): ['0', '1'],
        ('glue', 'mrpc'): ['0', '1'],
        ('glue', 'qqp'): ['0', '1'],
        ('glue', 'sst2'): ['0', '1'],
        ('glue', 'mnli'): ['entailment', 'neutral', 'contradiction'],
        ('glue', 'qnli'): ['entailment', 'not_entailment'],
        ('glue', 'rte'): ['entailment', 'not_entailment'],
    }

    for path_map_key in path_map:
      candidate_dataset_path, candidate_dataset_name, split = path_map_key
      if (candidate_dataset_path != config.dataset_path
          or candidate_dataset_name != config.dataset_name):
        continue

      predict_iter = data_pipeline.get_inputs(
          split=split, batch_size=config.eval_batch_size, training=False)
      predict_stats = predict_step(optimizer, predict_iter)
      idxs = predict_stats['idx']
      predictions = predict_stats['prediction']

      tsv_path = os.path.join(
        output_dir, path_map[config.dataset_path, config.dataset_name, split])
      with gfile.GFile(tsv_path, 'w') as f:
        f.write('index\tprediction\n')
        if is_regression_task:
          for idx, val in zip(idxs, predictions):
            f.write(f'{idx}\t{val:.06f}\n')
        else:
          label_set = label_sets[config.dataset_path, config.dataset_name]
          for idx, val in zip(idxs, predictions):
            f.write(f'{idx}\t{label_set[val]}\n')
      print('Wrote', tsv_path)


if __name__ == '__main__':
  app.run(main)
