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

"""Run masked LM/next sentence masked_lm pre-training for BERT."""

import datetime
import glob
import itertools
import json
import os

from absl import app
from absl import flags
from flax import nn
from flax import optim
import data
import import_weights
import modeling
import training
# from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow.io import gfile

from transformers import BertTokenizerFast

from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'output_dir', None,
    'The output directory where the model checkpoints will be written.')

config_flags.DEFINE_config_file(
  'config', None,
  'Hyperparameter configuration')


def get_output_dir(config):
  """Get output directory location."""
  del config
  output_dir = FLAGS.output_dir
  if output_dir is None:
    output_name = 'pretrain_{timestamp}'.format(
        timestamp=datetime.datetime.now().strftime('%Y%m%d_%H%M'),
    )
    output_dir = os.path.join('~', 'efficient_transformers', output_name)
    output_dir = os.path.expanduser(output_dir)
    print()
    print('No --output_dir specified')
    print('Using default output_dir:', output_dir, flush=True)
  return output_dir


def create_model(config):
  """Create a model, starting with a pre-trained checkpoint."""
  model_kwargs = dict(
      config=config.model,
  )
  model_def = modeling.BertForPreTraining.partial(**model_kwargs)
  if config.init_checkpoint:
    initial_params = import_weights.load_params(
        init_checkpoint=config.init_checkpoint,
        hidden_size=config.model.hidden_size,
        num_attention_heads=config.model.num_attention_heads,
        keep_masked_lm_head=True)
  else:
    with nn.stochastic(jax.random.PRNGKey(0)):
      _, initial_params = model_def.init_by_shape(
          jax.random.PRNGKey(0),
          [((1, config.max_seq_length), jnp.int32),
           ((1, config.max_seq_length), jnp.int32),
           ((1, config.max_seq_length), jnp.int32),
           ((1, config.max_predictions_per_seq), jnp.int32)],
          deterministic=True)
      def fixup_for_tpu(x, i=[0]):
        """HACK to fix incorrect param initialization on TPU."""
        if isinstance(x, jax.ShapeDtypeStruct):
          i[0] += 1
          if len(x.shape) == 2:
            return jnp.zeros(x.shape, x.dtype)
          else:
            return nn.linear.default_kernel_init(jax.random.PRNGKey(i[0]), x.shape, x.dtype)
        else:
          return x
      initial_params = jax.tree_map(fixup_for_tpu, initial_params)
  model = nn.Model(model_def, initial_params)
  return model


def create_optimizer(config, model):
  if config.optimizer == 'adam':
    optimizer_cls = optim.Adam
  elif config.optimizer == 'lamb':
    optimizer_cls = optim.LAMB
  else:
    raise ValueError('Unsupported value for optimizer: {config.optimizer}')
  common_kwargs = dict(
    learning_rate=config.learning_rate,
    beta1=0.9,
    beta2=0.999,
    eps=1e-6,
  )
  optimizer_decay_def = optimizer_cls(
    weight_decay=0.01, **common_kwargs)
  optimizer_no_decay_def = optimizer_cls(
    weight_decay=0.0, **common_kwargs)
  decay = optim.ModelParamTraversal(lambda path, _: 'bias' not in path)
  no_decay = optim.ModelParamTraversal(lambda path, _: 'bias' in path)
  optimizer_def = optim.MultiOptimizer(
    (decay, optimizer_decay_def), (no_decay, optimizer_no_decay_def))
  optimizer = optimizer_def.create(model)
  return optimizer


def compute_pretraining_loss_and_metrics(model, batch, rng):
  """Compute cross-entropy loss for classification tasks."""
  with nn.stochastic(rng):
    metrics = model(
        batch['input_ids'],
        (batch['input_ids'] > 0).astype(np.int32),
        batch['token_type_ids'],
        batch['masked_lm_positions'],
        batch['masked_lm_ids'],
        batch['masked_lm_weights'],
        batch['next_sentence_label'])
  return metrics['loss'], metrics


def compute_pretraining_stats(model, batch):
  """Used for computing eval metrics during pre-training."""
  with nn.stochastic(jax.random.PRNGKey(0)):
    masked_lm_logits, next_sentence_logits = model(
        batch['input_ids'],
        (batch['input_ids'] > 0).astype(np.int32),
        batch['token_type_ids'],
        batch['masked_lm_positions'],
        deterministic=True)
    stats = model.compute_metrics(
        masked_lm_logits, next_sentence_logits,
        batch['masked_lm_ids'],
        batch['masked_lm_weights'],
        batch['next_sentence_label'])

  masked_lm_correct = jnp.sum(
      (masked_lm_logits.argmax(-1) == batch['masked_lm_ids'].reshape((-1,))
       ) * batch['masked_lm_weights'].reshape((-1,)))
  next_sentence_labels = batch['next_sentence_label'].reshape((-1,))
  next_sentence_correct = jnp.sum(
      next_sentence_logits.argmax(-1) == next_sentence_labels)
  stats = {
      'masked_lm_correct': masked_lm_correct,
      'masked_lm_total': jnp.sum(batch['masked_lm_weights']),
      'next_sentence_correct': next_sentence_correct,
      'next_sentence_total': jnp.sum(jnp.ones_like(next_sentence_labels)),
      **stats
  }
  return stats


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  config = FLAGS.config

  model = create_model(config)
  optimizer = create_optimizer(config, model)
  del model  # don't keep a copy of the initial model

  output_dir = get_output_dir(config)
  gfile.makedirs(output_dir)

  # Restore from a local checkpoint, if one exists.
  optimizer = checkpoints.restore_checkpoint(output_dir, optimizer)
  start_step = int(optimizer.state[0].step)

  optimizer = optimizer.replicate()
  optimizer = training.harmonize_across_hosts(optimizer)

  os.environ['TOKENIZERS_PARALLELISM'] = 'true'
  tokenizer = BertTokenizerFast.from_pretrained(config.tokenizer)
  tokenizer.model_max_length = config.max_seq_length

  data_pipeline = data.PretrainingDataPipeline(
    glob.glob('cache/pretrain.*_of_*.tfrecord'),
    tokenizer,
    max_predictions_per_seq=config.max_predictions_per_seq)

  learning_rate_fn = training.create_learning_rate_scheduler(
      factors='constant * linear_warmup * linear_decay',
      base_learning_rate=config.learning_rate,
      warmup_steps=config.num_warmup_steps,
      steps_per_cycle=config.num_train_steps - config.num_warmup_steps,
  )

  train_history = training.TrainStateHistory(learning_rate_fn)
  train_state = train_history.initial_state()

  if config.do_train:
    train_iter = data_pipeline.get_inputs(
        batch_size=config.train_batch_size, training=True)
    train_step_fn = training.create_train_step(
        compute_pretraining_loss_and_metrics, clip_grad_norm=1.0)

    for step, batch in zip(range(start_step, config.num_train_steps),
                           train_iter):
      optimizer, train_state = train_step_fn(optimizer, batch, train_state)
      if jax.host_id() == 0 and (step % config.save_checkpoints_steps == 0
                                 or step == config.num_train_steps-1):
        checkpoints.save_checkpoint(output_dir,
                                    optimizer.unreplicate(),
                                    step)
        config_path = os.path.join(output_dir, 'config.json')
        if not os.path.exists(config_path):
          with open(config_path, 'w') as f:
            json.dump({'model_type': 'bert', **config.model}, f)

  if config.do_eval:
    eval_iter = data_pipeline.get_inputs(batch_size=config.eval_batch_size)
    eval_iter = itertools.islice(eval_iter, config.max_eval_steps)
    eval_fn = training.create_eval_fn(
        compute_pretraining_stats, sample_feature_name='input_ids')
    eval_stats = eval_fn(optimizer, eval_iter)

    eval_metrics = {
        'loss': jnp.mean(eval_stats['loss']),
        'masked_lm_loss': jnp.mean(eval_stats['masked_lm_loss']),
        'next_sentence_loss': jnp.mean(eval_stats['next_sentence_loss']),
        'masked_lm_accuracy': jnp.sum(
            eval_stats['masked_lm_correct']
            ) / jnp.sum(eval_stats['masked_lm_total']),
        'next_sentence_accuracy': jnp.sum(
            eval_stats['next_sentence_correct']
            ) / jnp.sum(eval_stats['next_sentence_total']),
    }

    eval_results = []
    for name, val in sorted(eval_metrics.items()):
      line = f'{name} = {val:.06f}'
      print(line, flush=True)
      eval_results.append(line)

    eval_results_path = os.path.join(output_dir, 'eval_results.txt')
    with gfile.GFile(eval_results_path, 'w') as f:
      for line in eval_results:
        f.write(line + '\n')


if __name__ == '__main__':
  app.run(main)
