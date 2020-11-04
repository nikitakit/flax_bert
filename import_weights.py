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

"""Code for loading weights via HuggingFace PyTorch checkpoints."""

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

import transformers
import os
import glob

from transformers import file_utils
from transformers.file_utils import WEIGHTS_NAME, CONFIG_NAME, is_remote_url, cached_path, hf_bucket_url

import torch

import flax.training.checkpoints
import flax.serialization


def get_pretrained_state_dict(pretrained_model_name_or_path, *model_args, **kwargs):
  """Get PyTorch state dict via HuggingFace transformers library."""
  config = kwargs.pop("config", None)
  state_dict = kwargs.pop("state_dict", None)
  cache_dir = kwargs.pop("cache_dir", None)
  # from_tf = kwargs.pop("from_tf", False)
  force_download = kwargs.pop("force_download", False)
  resume_download = kwargs.pop("resume_download", False)
  proxies = kwargs.pop("proxies", None)
  output_loading_info = kwargs.pop("output_loading_info", False)
  local_files_only = kwargs.pop("local_files_only", False)
  use_cdn = kwargs.pop("use_cdn", True)
  mirror = kwargs.pop("mirror", None)

  if pretrained_model_name_or_path is not None:
    if os.path.isdir(pretrained_model_name_or_path):
      if os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
        # Load from a PyTorch checkpoint
        archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
      else:
        raise EnvironmentError(
          "Error no file named {} found in directory {}".format(
            WEIGHTS_NAME,
            pretrained_model_name_or_path,
          )
        )
    elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
      archive_file = pretrained_model_name_or_path
    elif os.path.isfile(pretrained_model_name_or_path + ".index"):
      assert False, "Loading TensorFlow checkpoints is not supported"
    else:
      archive_file = hf_bucket_url(
        pretrained_model_name_or_path,
        filename=WEIGHTS_NAME,
        use_cdn=use_cdn,
        mirror=mirror,
      )

    try:
      # Load from URL or cache if already cached
      resolved_archive_file = cached_path(
        archive_file,
        cache_dir=cache_dir,
        force_download=force_download,
        proxies=proxies,
        resume_download=resume_download,
        local_files_only=local_files_only,
      )
      if resolved_archive_file is None:
        raise EnvironmentError
    except EnvironmentError:
      msg = (
        f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
        f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
        f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named {WEIGHTS_NAME}.\n\n"
      )
      raise EnvironmentError(msg)

    if resolved_archive_file == archive_file:
      print("loading weights file {}".format(archive_file))
    else:
      print("loading weights file {} from cache at {}".format(archive_file, resolved_archive_file))
  else:
    resolved_archive_file = None


  if state_dict is None:
    try:
      state_dict = torch.load(resolved_archive_file, map_location="cpu")
    except Exception:
      raise OSError("Unable to load weights from pytorch checkpoint file.")
  return state_dict


def load_params_from_hf(init_checkpoint, hidden_size, num_attention_heads):
  pt_params = get_pretrained_state_dict(init_checkpoint)
  jax_params = {}
  # mapping between HuggingFace PyTorch BERT and JAX model
  pt_key_to_jax_key = [
    # Output heads
    ('cls.seq_relationship', 'classification'),
    ('cls.predictions.transform.LayerNorm', 'predictions_transform_layernorm'),
    ('cls.predictions.transform.dense', 'predictions_transform_dense'),
    ('cls.predictions.bias', 'predictions_output.bias'),
    ('cls.predictions.decoder.weight', 'UNUSED'),
    ('cls.predictions.decoder.bias', 'UNUSED'),
    # Embeddings
    ('embeddings.position_ids', 'UNUSED'),
    ('embeddings.word_embeddings.weight', 'word_embeddings.embedding'),
    ('embeddings.token_type_embeddings.weight', 'type_embeddings.embedding'),
    ('embeddings.position_embeddings.weight', 'position_embeddings.embedding'),
    ('embeddings.LayerNorm', 'embeddings_layer_norm'),
    # Pooler
    ('pooler.dense.', 'pooler.'),
    # Layers
    ('bert.encoder.layer.', 'bert.encoder_layer_'),
    # ('bert/encoder/layer_', 'bert/encoder_layer_'),
    ('attention.self', 'self_attention.attn'),
    ('attention.output.dense', 'self_attention.attn.output'),
    ('attention.output.LayerNorm', 'self_attention_layer_norm'),
    ('output.LayerNorm', 'output_layer_norm'),
    ('intermediate.dense', 'feed_forward.intermediate'),
    ('output.dense', 'feed_forward.output'),
    # Parameter names
    ('weight', 'kernel'),
    ('beta', 'bias'),
    ('gamma', 'scale'),
    ('layer_norm.kernel', 'layer_norm.scale'),
    ('layernorm.kernel', 'layernorm.scale'),
    ]
  pt_keys_to_transpose = (
      "dense.weight",
      "attention.self.query",
      "attention.self.key",
      "attention.self.value"
      )
  for pt_key, val in pt_params.items():
    jax_key = pt_key
    for pt_name, jax_name in pt_key_to_jax_key:
      jax_key = jax_key.replace(pt_name, jax_name)

    if 'UNUSED' in jax_key:
        continue

    if any([x in pt_key for x in pt_keys_to_transpose]):
        val = val.T
    val = np.asarray(val)

    # Reshape kernels if necessary
    reshape_params = ['key', 'query', 'value']
    for key in reshape_params:
      if f'self_attention.attn.{key}.kernel' in jax_key:
        val = np.swapaxes(
            val.reshape((hidden_size, num_attention_heads, -1)), 0, 1)
      elif f'self_attention.attn.{key}.bias' in jax_key:
        val = val.reshape((num_attention_heads, -1))
    if 'self_attention.attn.output.kernel' in jax_key:
      val = val.reshape((num_attention_heads, -1, hidden_size))
    elif 'self_attention.attn.output.bias' in jax_key:
      # The multihead attention implementation we use creates a bias vector for
      # each head, even though this is highly redundant.
      val = np.stack(
          [val] + [np.zeros_like(val)] * (num_attention_heads - 1), axis=0)

    jax_params[jax_key] = val

  # jax position embedding kernel has additional dimension
  pos_embedding = jax_params[
      'bert.position_embeddings.embedding']
  jax_params[
      'bert.position_embeddings.embedding'] = pos_embedding[
          np.newaxis, ...]

  # this layer doesn't have parameters, but key is required to be present
  jax_params['GatherIndexes_0'] = {}

  # convert flat param dict into nested dict using `/` as delimeter
  outer_dict = {}
  for key, val in jax_params.items():
    tokens = key.split('.')
    inner_dict = outer_dict
    # each token except the very last should add a layer to the nested dict
    for token in tokens[:-1]:
      if token not in inner_dict:
        inner_dict[token] = {}
      inner_dict = inner_dict[token]
    inner_dict[tokens[-1]] = val

  if 'global_step' in outer_dict:
    del outer_dict['global_step']

  return outer_dict


def load_params(init_checkpoint, hidden_size, num_attention_heads,
                num_classes=None, keep_masked_lm_head=False):
  params = None
  if os.path.isdir(init_checkpoint):
    prefix = 'checkpoint_'
    glob_path = os.path.join(init_checkpoint, f'{prefix}*')
    checkpoint_files = flax.training.checkpoints.natural_sort(glob.glob(glob_path))

    ckpt_tmp_path = flax.training.checkpoints._checkpoint_path(init_checkpoint, 'tmp', prefix)
    checkpoint_files = [f for f in checkpoint_files if f != ckpt_tmp_path]
    if checkpoint_files:
      ckpt_path = checkpoint_files[-1]
      with open(os.path.expanduser(ckpt_path), 'rb') as f:
        params = flax.serialization.msgpack_restore(f.read())['target']['params']
  if params is None:
    params = load_params_from_hf(init_checkpoint, hidden_size, num_attention_heads)

  if not keep_masked_lm_head:
    del params['predictions_output']
    del params['predictions_transform_dense']
    del params['predictions_transform_layernorm']

  if num_classes is not None:
    # Re-initialize the output head
    output_projection = params['classification']
    output_projection['kernel'] = np.random.normal(
        scale=0.02,
        size=(num_classes, output_projection['kernel'].shape[1])).astype(
            output_projection['kernel'].dtype)
    output_projection['bias'] = np.zeros(
        num_classes, dtype=output_projection['bias'].dtype)

  # For some reason, using numpy arrays as weights doesn't cause a type error,
  # but instead leads to a shape discrepancy in some of the layers!
  params = jax.tree_map(jnp.asarray, params)
  return params
