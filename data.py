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

"""Input pipelines."""

import numpy as np
import random

import multiprocessing
import jax
import tensorflow as tf


class DataPipeline:
  """Base class for input pipelines based on tf.data (and optionally tfds).

  Subclasses must override the following methods:
    get_tf_dataset(self, batch_size, split, training)
    process_batch(self, batch)
  
  Use get_inputs(...) to generate an iterator over batches of data, represented
  as python dicts mapping from strings to containing numpy arrays.
  """
  def __init__(self, num_workers=0, prefetch_amount=None):
    """Construct a DataPipeline.

    Args:
      num_workers: Number of worker processes to use when calling
          self.process_batch. The default value of 0 performs python
          post-processing in the main thread, but this can be a performance
          bottleneck.
      prefetch_amount: Number of batches to process in advance.
    """
    self.num_workers = num_workers
    if prefetch_amount is None:
      self.prefetch_amount = num_workers * 2
    else:
      assert prefetch_amount >= num_workers
      self.prefetch_amount = prefetch_amount

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], "GPU")

  def get_tf_dataset(self, batch_size, split, training):
    raise NotImplementedError(
      "DataPipeline subclasses must define a get_tf_dataset function.")

  def process_batch(self, batch):
    raise NotImplementedError(
      "DataPipeline subclasses must define a process_batch function.")

  def get_inputs(self, batch_size, split=None, training=False):
    """Returns an iteractor over batches of examples.

    The behavior of this method should be equivalent to:
        for example in dataset.as_numpy_iterator():
          yield self.process_batch(example)

    However, running process_batch in the main thread may create a performance
    bottleneck, depending on the amount of processing needed. The `num_workers`
    and `prefetch_amount` constructor arguments allow using multiprocessing to
    move the dataset out of the main thread.
    """
    dataset = self.get_tf_dataset(batch_size, split, training)

    if self.num_workers == 0:
      for example in dataset.as_numpy_iterator():
        yield self.process_batch(example)
      return
    
    dummy_in = next(dataset.as_numpy_iterator())
    treedef_in = jax.tree_structure(dummy_in)
    dummy_out = self.process_batch(dummy_in)
    leaves_out, treedef_out = jax.tree_flatten(dummy_out)
    dtypes_out = [leaf.dtype for leaf in leaves_out]

    with multiprocessing.Pool(self.num_workers) as pool:
      dataset_head = dataset.take(self.prefetch_amount)
      dataset_tail = dataset.skip(self.prefetch_amount)
      prefetch = [
        pool.apply_async(self.process_batch, (inp,))
        for inp in dataset_head.as_numpy_iterator()
      ]

      def outer(in_tf):
        leaves_in_tf, treedef_in2 = jax.tree_flatten(in_tf)
        assert treedef_in == treedef_in2
        def inner(*leaves_in_np):
          nonlocal prefetch
          in_np = jax.tree_unflatten(treedef_in, leaves_in_np)
          prefetch.append(pool.apply_async(self.process_batch, (in_np,)))
          out_np = prefetch.pop(0).get()
          leaves_out_np, treedef_out2 = jax.tree_flatten(out_np)
          assert treedef_out == treedef_out2
          return leaves_out_np
        leaves_out_tf = tf.numpy_function(inner, leaves_in_tf, dtypes_out)
        out_tf = jax.tree_unflatten(treedef_out, leaves_out_tf)
        return out_tf

      yield from dataset_tail.map(outer).as_numpy_iterator()
      for out_result in prefetch:
        yield out_result.get()


class ClassificationDataPipeline(DataPipeline):
  def __init__(self, builder_constructor, tokenizer):
    super().__init__(num_workers=0)

    # Tensorflow has been configured to use CPU only, so we are now ready to
    # start working with tfds.
    self.dataset_builder = builder_constructor()
    self.dataset_builder.download_and_prepare()
    self.tokenizer = tokenizer
    self.shuffle_buffer_size = 1024
    self.batch_shuffle_size = 128

    self.name_a, *names_other = [
      name for name, feature in self.dataset_builder.info.features.items()
      if feature.dtype=='string']
    assert len(names_other) <= 1, (
      'Only single sentences and sentence pairs allowed.')
    if names_other:
      self.name_b = names_other[0]
    else:
      self.name_b = None

  def get_tf_dataset(self, batch_size, split, training):
    d = self.dataset_builder.as_dataset(split=split, shuffle_files=training)
    if training:
      d = d.repeat()
    if training and self.shuffle_buffer_size is not None:
      d = d.shuffle(self.shuffle_buffer_size)
    d = d.batch(batch_size)
    if training and self.batch_shuffle_size is not None:
      d = d.shuffle(self.batch_shuffle_size)
    return d

  def process_batch(self, batch):
    text_a = [x.decode('utf-8') for x in batch[self.name_a].tolist()]
    if self.name_b is not None:
      text_b = [x.decode('utf-8') for x in batch[self.name_b].tolist()]
    else:
      text_b = None
    res = self.tokenizer(
      text_a, text_b,
      truncation=True,
      padding='max_length',
      max_length=self.tokenizer.model_max_length,
      return_tensors='np',
      return_attention_mask=False,
      )
    res = dict(res)  # Custom dict subclass is not a pytree
    res['idx'] = batch['idx']
    res['label'] = batch['label']
    return res


class PretrainingDataPipeline(DataPipeline):
  def __init__(self, input_files, tokenizer, max_predictions_per_seq=20,
      num_workers=4, num_cpu_threads=32):
    super().__init__(num_workers=4)
    self.input_files = input_files
    self.tokenizer = tokenizer
    self.max_predictions_per_seq = max_predictions_per_seq
    self.num_cpu_threads = num_cpu_threads

    self.ignore_ids = np.array([
      self.tokenizer.cls_token_id,
      self.tokenizer.sep_token_id,
      self.tokenizer.pad_token_id
      ], dtype=np.int64)[:,None,None]

  def decode_record(self, record):
    example = tf.io.parse_single_example(
      record,
      features={
        "input_ids": tf.io.RaggedFeature(tf.int64),
      }
    )
    example['input_ids'] = tf.pad(
      example['input_ids'],
      ((0, self.tokenizer.model_max_length - tf.shape(example['input_ids'])[0]),)
    )
    example['input_ids'].set_shape((self.tokenizer.model_max_length,))
    return example

  def get_tf_dataset(self, batch_size, split, training):
    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(self.input_files))
      d = d.repeat()
      d = d.shuffle(buffer_size=len(self.input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(self.num_cpu_threads, len(self.input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.interleave(
        tf.data.TFRecordDataset,
        cycle_length,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=not training) 
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(self.input_files)

    d = d.map(self.decode_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    d = d.batch(batch_size, drop_remainder=True)
    return d

  def process_batch(self, batch):
    batch_size = batch['input_ids'].shape[0]
    batch['masked_lm_positions'] = np.zeros(
      (batch_size, self.max_predictions_per_seq), dtype=np.int64)
    batch['masked_lm_ids'] = np.zeros(
      (batch_size, self.max_predictions_per_seq), dtype=np.int64)
    batch['masked_lm_weights'] = np.zeros(
      (batch_size, self.max_predictions_per_seq), dtype=np.float32)

    # Sentence Order Prediction task
    batch['next_sentence_label'] = np.random.randint(0, 2, batch_size, dtype=np.int64)
    segments = np.cumsum(
      batch['input_ids'][:,::-1] == self.tokenizer.sep_token_id, axis=-1)[:,::-1]
    segments[:, 0] = 1
    swapped_segments = np.argsort(
      np.where(segments==1, -3, -segments), axis=-1, kind='stable')
    swapped_input_ids = np.take_along_axis(
      batch['input_ids'], swapped_segments, axis=-1)
    batch['input_ids'] = np.where(
      batch['next_sentence_label'][:, None], swapped_input_ids, batch['input_ids'])

    # Token type ids
    batch['token_type_ids'] = np.cumsum(
      batch['input_ids'][:, ::-1] == self.tokenizer.sep_token_id,
      axis=-1)[:, ::-1] % 2

    # Masked LM task
    prediction_mask = np.all(batch['input_ids'] != self.ignore_ids, axis=0)
    num_tokens = np.sum(batch['input_ids'] != self.tokenizer.pad_token_id, axis=-1)
    for i in range(batch_size):
      cand_indexes = np.arange(
          prediction_mask.shape[1], dtype=np.int32)[prediction_mask[i]]
      num_to_predict = min(
          self.max_predictions_per_seq, max(1, int(num_tokens[i] * 0.15)))

      masked_lm_positions = np.random.choice(
          cand_indexes, num_to_predict, replace=False)
      masked_lm_positions = np.sort(masked_lm_positions)
      masked_lm_ids = batch['input_ids'][i, masked_lm_positions]
      batch['masked_lm_positions'][i, :num_to_predict] = masked_lm_positions
      batch['masked_lm_ids'][i, :num_to_predict] = masked_lm_ids
      batch['masked_lm_weights'][i, :num_to_predict] = 1.0

    do_predict = prediction_mask[
      np.arange(batch_size)[:, None], batch['masked_lm_positions']]
    r = np.random.random(batch['masked_lm_ids'].shape)
    keep_original = (r < 0.1) | ~do_predict
    replace_with_mask = (r < 0.9)

    batch['input_ids'][np.arange(batch_size)[:, None], batch['masked_lm_positions']
      ] = np.where(
        keep_original,
        # 10% of the time, keep original
        batch['input_ids'][
          np.arange(batch_size)[:, None], batch['masked_lm_positions']],
        np.where(replace_with_mask,
          # 80% of the time, replace with [MASK]
          self.tokenizer.mask_token_id,
          # 10% of the time, replace with random word
          np.random.randint(
            0, self.tokenizer.vocab_size, batch['masked_lm_ids'].shape)))

    return batch
