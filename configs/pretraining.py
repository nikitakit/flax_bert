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

import ml_collections

def get_config(config_string=''):
  if config_string:
    import transformers
    hf_config = transformers.AutoConfig.from_pretrained(config_string)
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
  else:
    model_config = ml_collections.ConfigDict({
      'vocab_size': 30522,
      'hidden_size': 768,
      'num_hidden_layers': 12,
      'num_attention_heads': 12,
      'hidden_act': 'gelu',
      'intermediate_size': 3072,
      'hidden_dropout_prob': 0.1,
      'attention_probs_dropout_prob': 0.1,
      'max_position_embeddings': 512,
      'type_vocab_size': 2,
      # 'initializer_range': 0.02,
      # 'layer_norm_eps': 1e-12,
    })

  config = ml_collections.ConfigDict({
    # Configuration for the model
    'model': model_config,
    # Initial checkpoint (e.g. from a pre-trained BERT model).
    'init_checkpoint': config_string,
    # Pre-trained tokenizer
    'tokenizer': config_string,
    # Whether to run training.
    'do_train': True,
    # Whether to run eval.
    'do_eval': True,
    # Total batch size for training.
    'train_batch_size': 512,
    # Total batch size for eval.
    'eval_batch_size': 64,
    # Optimizer: either 'adam' or 'lamb
    'optimizer': 'lamb',
    # The base learning rate for Adam or LAMB.
    'learning_rate': 6.25e-4,
    # Number of training steps.
    'num_train_steps': 500000,
    # Number of warmup steps.
    'num_warmup_steps': 3125,
    # The maximum total input sequence length after tokenization.
    # Sequences longer than this will be truncated, and sequences shorter
    # than this will be padded. 
    'max_seq_length': 128,
    # Maximum number of masked LM predictions per sequence.
    'max_predictions_per_seq': 20,
    # How often to save the model checkpoint.
    'save_checkpoints_steps': 10000,
    # Maximum number of eval steps.
    'max_eval_steps': 100,
  })    

  return config