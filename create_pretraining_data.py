import collections
import glob
import multiprocessing
import os
import random

import tensorflow as tf
import transformers


def documents_from_filenames(files):
  """Yields documents one at a time, as lists of strings."""
  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.
  document = []
  for input_file in files:
    with open(input_file, "r", encoding='utf-8', errors='ignore') as reader:
      while True:
        line = reader.readline()
        if not line:
          break
        line = line.strip()

        # Empty lines are used as document delimiters
        if not line:
          yield document
          document = []
        else:
          document.append(line)
  
  if document:
    yield document


def examples_from_document(document, rng, tokenizer, short_seq_prob):
  max_seq_length = tokenizer.model_max_length
  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  document = [
    tokenizer.encode(sentence, add_special_tokens=False)
    for sentence in document
  ]

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.
  target_seq_length = max_num_tokens
  if rng.random() < short_seq_prob:
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  current_chunk = []
  current_length = 0
  for i, segment in enumerate(document):
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
        if len(current_chunk) == 1:
          # The current chunk consists of only one sentence, so we can't create
          # a two-segment training example from it. BERT just samples a second
          # segment from another random document, but we're using a sentence
          # order prediction (SOP) task to avoid looking across documents.
          # ALBERT actually matches BERT in this corner case by switching from
          # its usual SOP task to BERT's NSP.
          # TODO(kitaev): maybe don't just discard the sentence here?
          continue
        else:
          for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])

        tokens_a, tokens_b, _ = tokenizer.truncate_sequences(
          tokens_a, tokens_b, max(0, len(tokens_a) + len(tokens_b) - max_num_tokens))
        instance = tokenizer.prepare_for_model(
            tokens_a, tokens_b,
            return_token_type_ids=False,
            return_attention_mask=False,
            )
        # Don't add instances that consist entirely of UNK tokens
        if any(token not in tokenizer.all_special_ids for token in instance['input_ids']):
          yield instance
      current_chunk = []
      current_length = 0


class WorkerData:
  @classmethod
  def initializer(cls, d):
    cls.d = d

  @classmethod
  def serialized_examples_from_document(cls, document):
    examples = examples_from_document(
      document,
      cls.d['rng'], cls.d['tokenizer'], short_seq_prob=cls.d['short_seq_prob'])

    res = []
    for example in examples:
      features = collections.OrderedDict()
      features['input_ids'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=example['input_ids']))
      tf_example = tf.train.Example(features=tf.train.Features(feature=features))
      res.append(tf_example.SerializeToString())
    return res


def write_examples(input_filenames, output_prefix, num_workers, num_shards,
    tokenizer_name_or_path, max_seq_length, short_seq_prob):
  tokenizer = transformers.BertTokenizerFast.from_pretrained(tokenizer_name_or_path)
  tokenizer.model_max_length = max_seq_length

  def gen():
    documents = documents_from_filenames(input_filenames)
    with multiprocessing.Pool(
        num_workers,
        WorkerData.initializer, (dict(
              tokenizer=tokenizer,
              short_seq_prob=short_seq_prob,
              rng=random.Random(0),
            ),)) as pool:
      for examples in pool.imap_unordered(
          WorkerData.serialized_examples_from_document, documents, chunksize=8):
        yield from examples

  d = tf.data.Dataset.from_generator(gen, tf.string)
  d = d.shuffle(100000000, seed=0)
  def reduce_func(key, dataset):
    filename = tf.strings.join([
      output_prefix,
      ".",
      tf.strings.as_string(key, width=5, fill='0'),
      "_of_",
      tf.strings.as_string(num_shards, width=5, fill='0'),
      ".tfrecord",
      ])
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(dataset.map(lambda _, x: x))
    return tf.data.Dataset.from_tensors(filename)

  d = d.enumerate()
  d = d.apply(tf.data.experimental.group_by_window(
    lambda i, _: i % num_shards, reduce_func, tf.int64.max
  ))
  # Iterating the dataset causes the files to be written
  files_written = list(d)
  assert len(files_written) == num_shards

if __name__ == "__main__":
  tf.config.experimental.set_visible_devices([], "GPU")
  if not os.path.exists('cache'):
    os.makedirs('cache')
  write_examples(
    glob.glob('cleanup_scripts/results/part-?????-of-?????'),
    'cache/pretrain',
    num_workers=55,
    num_shards=500,
    tokenizer_name_or_path='bert-base-uncased',
    max_seq_length=128,
    short_seq_prob=0.1,
  )
