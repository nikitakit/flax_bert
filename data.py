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

"""Huggingface input pipelines."""

import datasets
import numpy as np
import random
import torch


class DataPipeline:
  def __init__(self, dataset):
    self.dataset = dataset

  def get_inputs(self, batch_size, split=None, training=False):
    dataloader = torch.utils.data.DataLoader(
      self.dataset if split is None else self.dataset[split],
      collate_fn=self.collate,
      batch_size=batch_size,
      drop_last=training,
      shuffle=training,
      num_workers=64,
      )
    if training:
      while True:
        for batch in iter(dataloader):
          yield dict(batch)  # The dict-like types from huggingface datasets are not pytrees
    else:
      for batch in iter(dataloader):
        yield dict(batch)  # The dict-like types from huggingface datasets are not pytrees

  def collate(self, examples):
    raise NotImplementedError("DataPipeline subclasess must define a collate function.")


class ClassificationDataPipeline(DataPipeline):
  def __init__(self, dataset, tokenizer):
    self.tokenizer = tokenizer

    if isinstance(dataset, dict):
      single_split = dataset['train']
    else:
      single_split = dataset

    name_a, *names_other = [
      name for name, feature in single_split.features.items()
      if feature.dtype=='string']
    assert len(names_other) <= 1, (
      'Only single sentences and sentence pairs allowed.')
    if names_other:
      name_b = names_other[0]
      tokenize = lambda example: self.tokenizer(
        example[name_a], example[name_b], truncation=True)
    else:
      tokenize = lambda example: self.tokenizer(
        example[name_a], truncation=True)
    mapped_dataset = dataset.map(tokenize, batched=True)
    mapped_dataset.set_format('numpy', columns=[
      'idx', 'input_ids', 'token_type_ids', 'attention_mask', 'label'])
    super().__init__(mapped_dataset)

  def collate(self, examples):
    return self.tokenizer.pad(
      examples,
      padding='max_length',
      max_length=self.tokenizer.model_max_length,
      return_tensors='np',
      )


class PretrainingDataPipeline(DataPipeline):
  FEATURE_NAMES = [
    'attention_mask',
    'input_ids',
    'token_type_ids',
    'next_sentence_label'
    ]

  def __init__(self, dataset, tokenizer,
      short_seq_prob=0.1,
      max_predictions_per_seq=80):
    self.tokenizer = tokenizer
    self.short_seq_prob = short_seq_prob
    self.max_predictions_per_seq = max_predictions_per_seq

    cache_file_name=(
      f"cache/"
      f"pretrain_new_l{self.tokenizer.model_max_length}"
      f"_s{self.short_seq_prob}"
      f"_p{self.max_predictions_per_seq}"
      f".arrow")

    mapped_dataset = dataset.map(
      self.examples_from_documents, batched=True,
      remove_columns=dataset.column_names,
      cache_file_name=cache_file_name,
      num_proc=32,
      features=datasets.Features(
        attention_mask=datasets.features.Sequence(
          datasets.features.Value(dtype='int8'), length=-1),
        input_ids=datasets.features.Sequence(
          datasets.features.Value(dtype='int16'), length=-1),
        token_type_ids=datasets.features.Sequence(
          datasets.features.Value(dtype='int8'), length=-1),
        next_sentence_label=datasets.features.Value(dtype='int8'),
      ),
      fn_kwargs=dict(
        rng=random.Random(0),
      ))
    mapped_dataset.set_format('numpy')

    super().__init__(mapped_dataset)

  def collate(self, examples):
    examples = self.tokenizer.pad(
      examples,
      padding='max_length',
      max_length=self.tokenizer.model_max_length,
      return_tensors='np',
      )

    ignore_ids = np.array([
        self.tokenizer.cls_token_id,
        self.tokenizer.sep_token_id,
        self.tokenizer.pad_token_id
        ], dtype=np.int64)[:, None]

    batch_size = examples['input_ids'].shape[0]
    examples['input_ids'] = examples['input_ids'].copy()
    examples['masked_lm_positions'] = np.zeros((batch_size, self.max_predictions_per_seq), dtype=np.int64)
    examples['masked_lm_ids'] = np.zeros((batch_size, self.max_predictions_per_seq), dtype=np.int64)
    examples['masked_lm_weights'] = np.zeros((batch_size, self.max_predictions_per_seq), dtype=np.float32)
    for i in range(batch_size):
      prediction_mask = np.all(examples['input_ids'][i] != ignore_ids, axis=0)
      num_tokens = np.sum(examples['attention_mask'][i]).item()
      cand_indexes = np.arange(
          prediction_mask.shape[0], dtype=np.int32)[prediction_mask]
      num_to_predict = min(
          self.max_predictions_per_seq, max(1, int(num_tokens * 0.15)))

      masked_lm_positions = np.random.choice(
          cand_indexes, num_to_predict, replace=False)
      masked_lm_positions = np.sort(masked_lm_positions)
      input_ids = examples['input_ids'][i].copy()
      masked_lm_ids = input_ids[masked_lm_positions]

      input_ids[masked_lm_positions] = np.where(
        np.random.random(len(masked_lm_ids)) < 0.8,
        # 80% of the time, replace with [MASK]
        self.tokenizer.mask_token_id,
        np.where(np.random.random(len(masked_lm_ids)) < 0.5,
          # 10% of the time, keep original
          masked_lm_ids,
          # 10% of the time, replace with random word
          np.random.randint(0, self.tokenizer.vocab_size, masked_lm_ids.shape)))
      examples['input_ids'][i, :] = input_ids

      examples['masked_lm_positions'][i, :num_to_predict] = masked_lm_positions
      examples['masked_lm_ids'][i, :num_to_predict] = masked_lm_ids
      examples['masked_lm_weights'][i, :num_to_predict] = 1.0

    return examples

  def examples_from_documents(self, documents, rng):
    max_seq_length = self.tokenizer.model_max_length
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    instances = []
    for text in documents['document']:
      document = [
        self.tokenizer.encode(line, add_special_tokens=False)
        for line in text
      ]

      # We *usually* want to fill up the entire sequence since we are padding
      # to `max_seq_length` anyways, so short sequences are generally wasted
      # computation. However, we *sometimes*
      # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
      # sequences to minimize the mismatch between pre-training and fine-tuning.
      # The `target_seq_length` is just a rough target however, whereas
      # `max_seq_length` is a hard limit.
      target_seq_length = max_num_tokens
      if rng.random() < self.short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

      # We DON'T just concatenate all of the tokens from a document into a long
      # sequence and choose an arbitrary split point because this would make the
      # next sentence prediction task too easy. Instead, we split the input into
      # segments "A" and "B" based on the actual "sentences" provided by the user
      # input.
      current_chunk = []
      current_length = 0
      i = 0
      while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
          if current_chunk:
            # `a_end` is how many segments from `current_chunk` go into the `A`
            # (first) sentence.
            a_end = 1
            if len(current_chunk) >= 2:
              a_end = rng.randint(1, len(current_chunk) - 1)

            tokens_a = []
            for j in range(a_end):
              tokens_a.extend(current_chunk[j])

            tokens_b = []
            # Random next
            is_random_next = False
            if len(current_chunk) == 1:
              continue  # XXX
            elif rng.random() < 0.5:
              is_random_next = True
              for j in range(a_end, len(current_chunk)):
                tokens_b.extend(current_chunk[j])
              # Note(mingdachen): in this case, we just swap tokens_a and tokens_b
              tokens_a, tokens_b = tokens_b, tokens_a
            # Actual next
            else:
              is_random_next = False
              for j in range(a_end, len(current_chunk)):
                tokens_b.extend(current_chunk[j])

            tokens_a, tokens_b, _ = self.tokenizer.truncate_sequences(
              tokens_a, tokens_b, max(0, len(tokens_a) + len(tokens_b) - max_num_tokens))
            instance = self.tokenizer.prepare_for_model(tokens_a, tokens_b)
            if any(token not in self.tokenizer.all_special_ids for token in instance['input_ids']):
              # Don't add instances that consist entirely of UNK tokens
              instances.append(instance)
              instance['next_sentence_label'] = int(is_random_next)
          current_chunk = []
          current_length = 0
        i += 1

    return {k: [instance[k] for instance in instances] for k in self.FEATURE_NAMES}


class PretrainingDataPipelineV1(DataPipeline):
  FEATURE_NAMES = [
    'attention_mask',
    'input_ids',
    'token_type_ids',
    'next_sentence_label'
    ]

  def __init__(self, dataset, tokenizer,
      short_seq_prob=0.1,
      max_predictions_per_seq=80):
    self.tokenizer = tokenizer
    self.short_seq_prob = short_seq_prob
    self.max_predictions_per_seq = max_predictions_per_seq

    cache_file_name=(
      f"cache/"
      f"pretrain_l{self.tokenizer.model_max_length}"
      f"_s{self.short_seq_prob}"
      f"_p{self.max_predictions_per_seq}"
      f".arrow")

    mapped_dataset = dataset.map(
      self.examples_from_documents, batched=True,
      remove_columns=dataset.column_names,
      cache_file_name=cache_file_name,
      num_proc=32,
      features=datasets.Features(
        attention_mask=datasets.features.Sequence(
          datasets.features.Value(dtype='int8'), length=-1),
        input_ids=datasets.features.Sequence(
          datasets.features.Value(dtype='int16'), length=-1),
        token_type_ids=datasets.features.Sequence(
          datasets.features.Value(dtype='int8'), length=-1),
        next_sentence_label=datasets.features.Value(dtype='int8'),
      ),
      fn_kwargs=dict(
        rng=random.Random(0),
      ))
    mapped_dataset.set_format('numpy')

    super().__init__(mapped_dataset)

  def collate(self, examples):
    examples = self.tokenizer.pad(
      examples,
      padding='max_length',
      max_length=self.tokenizer.model_max_length,
      return_tensors='np',
      )

    ignore_ids = np.array([
        self.tokenizer.cls_token_id,
        self.tokenizer.sep_token_id,
        self.tokenizer.pad_token_id
        ], dtype=np.int64)[:, None]

    batch_size = examples['input_ids'].shape[0]
    examples['input_ids'] = examples['input_ids'].copy()
    examples['masked_lm_positions'] = np.zeros((batch_size, self.max_predictions_per_seq), dtype=np.int64)
    examples['masked_lm_ids'] = np.zeros((batch_size, self.max_predictions_per_seq), dtype=np.int64)
    examples['masked_lm_weights'] = np.zeros((batch_size, self.max_predictions_per_seq), dtype=np.float32)
    for i in range(batch_size):
      prediction_mask = np.all(examples['input_ids'][i] != ignore_ids, axis=0)
      num_tokens = np.sum(examples['attention_mask'][i]).item()
      cand_indexes = np.arange(
          prediction_mask.shape[0], dtype=np.int32)[prediction_mask]
      num_to_predict = min(
          self.max_predictions_per_seq, max(1, int(num_tokens * 0.15)))

      masked_lm_positions = np.random.choice(
          cand_indexes, num_to_predict, replace=False)
      masked_lm_positions = np.sort(masked_lm_positions)
      input_ids = examples['input_ids'][i].copy()
      masked_lm_ids = input_ids[masked_lm_positions]

      input_ids[masked_lm_positions] = np.where(
        np.random.random(len(masked_lm_ids)) < 0.8,
        # 80% of the time, replace with [MASK]
        self.tokenizer.mask_token_id,
        np.where(np.random.random(len(masked_lm_ids)) < 0.5,
          # 10% of the time, keep original
          masked_lm_ids,
          # 10% of the time, replace with random word
          np.random.randint(0, self.tokenizer.vocab_size, masked_lm_ids.shape)))
      examples['input_ids'][i, :] = input_ids

      examples['masked_lm_positions'][i, :num_to_predict] = masked_lm_positions
      examples['masked_lm_ids'][i, :num_to_predict] = masked_lm_ids
      examples['masked_lm_weights'][i, :num_to_predict] = 1.0

    return examples

  def examples_from_documents(self, documents, rng):
    max_seq_length = self.tokenizer.model_max_length
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    instances = []
    for text in documents['text']:
      document = [
        self.tokenizer.encode(
          line.strip().replace("\n", " ").replace("()",""),
          add_special_tokens=False)
        for line in text.splitlines()
        if line.strip() and len(line) >= 80
      ]

      # We *usually* want to fill up the entire sequence since we are padding
      # to `max_seq_length` anyways, so short sequences are generally wasted
      # computation. However, we *sometimes*
      # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
      # sequences to minimize the mismatch between pre-training and fine-tuning.
      # The `target_seq_length` is just a rough target however, whereas
      # `max_seq_length` is a hard limit.
      target_seq_length = max_num_tokens
      if rng.random() < self.short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

      # We DON'T just concatenate all of the tokens from a document into a long
      # sequence and choose an arbitrary split point because this would make the
      # next sentence prediction task too easy. Instead, we split the input into
      # segments "A" and "B" based on the actual "sentences" provided by the user
      # input.
      current_chunk = []
      current_length = 0
      i = 0
      while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
          if current_chunk:
            # `a_end` is how many segments from `current_chunk` go into the `A`
            # (first) sentence.
            a_end = 1
            if len(current_chunk) >= 2:
              a_end = rng.randint(1, len(current_chunk) - 1)

            tokens_a = []
            for j in range(a_end):
              tokens_a.extend(current_chunk[j])

            tokens_b = []
            # Random next
            is_random_next = False
            if len(current_chunk) == 1:
              continue  # XXX
            elif rng.random() < 0.5:
              is_random_next = True
              for j in range(a_end, len(current_chunk)):
                tokens_b.extend(current_chunk[j])
              # Note(mingdachen): in this case, we just swap tokens_a and tokens_b
              tokens_a, tokens_b = tokens_b, tokens_a
            # Actual next
            else:
              is_random_next = False
              for j in range(a_end, len(current_chunk)):
                tokens_b.extend(current_chunk[j])

            tokens_a, tokens_b, _ = self.tokenizer.truncate_sequences(
              tokens_a, tokens_b, max(0, len(tokens_a) + len(tokens_b) - max_num_tokens))
            instance = self.tokenizer.prepare_for_model(tokens_a, tokens_b)
            if any(token not in self.tokenizer.all_special_ids for token in instance['input_ids']):
              # Don't add instances that consist entirely of UNK tokens
              instances.append(instance)
              instance['next_sentence_label'] = int(is_random_next)
          current_chunk = []
          current_length = 0
        i += 1

    return {k: [instance[k] for instance in instances] for k in self.FEATURE_NAMES}

