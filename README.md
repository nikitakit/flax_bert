# BERT in Flax

***NOTE: This implementation is work in progress!***

An implementation of BERT in JAX+Flax. Runs on CPU, GPU, and TPU.

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

To pre-train BERT, first follow the instructions in [`cleanup_scripts`](cleanup_scripts/README.md) to obtain text files of English Wikipedia.

Next, run `python create_pretraining_data.py` to convert the pre-training data to the tfrecord format. This will create a `cache` folder, which can be safely copied between machines (including different hosts in a multi-host setup) to save on processing time.

To begin pre-training, run a command such as:
```sh
python3 run_pretraining.py --config=configs/pretraining.py:bert-base-uncased --config.init_checkpoint=""
```

***Remember to specify a blank initial checkpoint as a hyperparameter. Otherwise, pre-training will resume from the already-trained bert-base-uncased checkpoint.***

For BERT base, our best hyperparameter setting thus far involves pre-training with global batch size 4096. To launch this training on `TPUv3-32` with 4 hosts, set up each host with the required environment variables and then run the following command:
```sh
python3 run_pretraining.py --config=configs/pretraining.py:bert-base-uncased --config.init_checkpoint="" --config.train_batch_size=1024 --config.learning_rate=0.0017677669529663688 
```

***Note: config.train_batch_size currently specifies batch size per host, not global batch size***

### Known differences from the original BERT code
* There is no publicly-available data pipeline for replicating BERT. Most notably, books corpus data is not publicly available. Pre-training on Wikipedia alone, like in this repo, may yield a slightly different quality of results.
* The next-sentence prediction (NSP) task from BERT is replaced with a sentence order prediction (SOP) from ALBERT. This is to make the data pipeline simpler and more parallelizable.
* BERT's Adam optimizer departs from the Adam paper in that it omits bias correction terms. This codebase uses Flax's implementation of Adam, which includes bias correction.
* There exist several variants of the LAMB optimizer, which differ in subtle details. This codebase uses Flax's implementation of LAMB, and we have not verified that our usage exactly matches any other implementation.
* Pre-training uses a fixed maximum sequence length of 128, and does not increase the sequence length to 512 for the last 10% of training.
* Random masking and sentence shuffling occurs each time a batch of examples is sampled during training, rather than a single time during the data generation step.
