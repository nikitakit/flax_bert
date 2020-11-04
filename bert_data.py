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

"""HuggingFace datasets loader script for BERT data text files."""

import os

import datasets


class BertDataset(datasets.GeneratorBasedBuilder):
  """Loads BERT training data."""

  def _info(self):
    return datasets.DatasetInfo(features=datasets.Features({
        'document': datasets.Sequence(datasets.Value("string"))
    }))

  def _split_generators(self, dl_manager):
    """We handle string, list and dicts in datafiles"""
    if not self.config.data_files:
      raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
    data_files = dl_manager.download_and_extract(self.config.data_files)
    if isinstance(data_files, (str, list, tuple)):
      files = data_files
      if isinstance(files, str):
        files = [files]
      return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"files": files})]
    splits = []
    for split_name, files in data_files.items():
      if isinstance(files, str):
        files = [files]
      splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
    return splits

  def _generate_examples(self, files):
    """ Yields examples. """
    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    id_ = 0
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
            yield id_, {
              'document': document
            }
            document = []
            id_ += 1
          else:
            document.append(line)
    
    if document:
      yield id_, {
        'document': document
      }