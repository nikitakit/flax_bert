# BERT in Flax

***NOTE: This implementation is work in progress!***

An implementation of BERT in JAX+Flax. Runs on CPU, GPU, and TPU (single-host only for now, e.g. `TPUv3-8`).

## Fine-tuning on GLUE

Sample command for fine-tuning on GLUE:

```sh
python3 run_classifier.py --config=configs/classifier.py:bert-base-uncased --config.dataset_name="cola"
```

The `dataset_name` should be one of: `cola`, `mrpc`, `qqp`, `sst2`, `stsb`, `mnli`, `qnli`, `rte`. WNLI is not supported because BERT accuracy on WNLI is below the baseline, unless a special training recipe is used.

A pre-trained BERT model hosted via the [transformers](https://github.com/huggingface/transformers) model hub may be selected by replacing `bert-base-uncased` with another name, e.g. `--config=configs/classifier.py:bert-large-uncased`.

You may also specify the tokenizer and checkpoint location separately. Use this method to load JAX checkpoints pre-trained using this codebase, as well as HuggingFace PyTorch checkpoints saved via `save_pretrained`:
```
--config=configs/classifier.py 
--config.tokenizer="bert-base-uncased"
--config.init_checkpoint="/path/to/model/dir"
```

## Pre-training

To pre-train BERT, first modify `run_pretraining.py` (around line 213) to point to the location of BERT training data in text format.

Then run a command like:
```sh
python3 run_pretraining.py --config=configs/pretraining.py:bert-base-uncased --config.init_checkpoint=""
```

***Remember to specify a blank initial checkpoint as a hyperparameter. Otherwise, pre-training will resume from the already-trained bert-base-uncased checkpoint.***

If you do not have your own BERT training data, you can edit  `run_pretraining.py` to comment out the data pipeline around lines 211-217, and uncomment lines 198-201 instead. This will switch to a reproducible data pipeline based on public data sources (Wikipedia), but we provide no guarantees that this pipeline will lead to good pre-trained models.

Known differences from the original BERT code:
* There is no publicly-available data pipeline for replicating BERT. I hope to eventually use the HuggingFace datasets library to provide an at least somewhat reasonable approximation.
* The next-sentence prediction (NSP) task from BERT is replaced with a sentence order prediction (SOP) from ALBERT. This is to make the data pipeline simpler and more parallelizable.
* BERT's Adam optimizer departs from the Adam paper in that it omits bias correction terms. This codebase uses Flax's implementation of Adam, which includes bias correction.
* Pre-training uses a fixed maximum sequence length of 128, and does not increase the sequence length to 512 for the last 10% of training.
