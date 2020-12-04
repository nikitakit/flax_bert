# Dataset processing scripts for Wikipedia

This file contains scripts for producing cleaned-up text dumps of Wikipedia.

First, download the [Wikipedia dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2) and extract the pages. The Wikipedia dump can be downloaded from this link in this directory, and should contain the following file: `enwiki-latest-pages-articles-multistream.xml.bz2`. Note that older database dumps are periodically deleted from the official Wikimedia downloads website, so we can't pin a specific version of Wikipedia without hosting a mirror of the full dataset.

Next, run `WikiExtractor.py` to extract the wiki pages from the XML. The generated wiki pages file will be stored as `<data dir>/LL/wiki_nn`; for example `<data dir>/AA/wiki_00`. Each file is ~1MB, and each sub directory has 100 files from `wiki_00` to `wiki_99`, except the last sub directory. For the dump around December 2020, the last file is `FL/wiki_09`.

Finally, run the clean up scripts in this directory. The following command will run the clean up steps, and put the results in `./results`: `./process_wiki.sh '<data dir>/*/wiki_??'`

After running the `process_wiki.sh` script, for the wiki dump around December 2020, there will be 500 files, named `part-00xxx-of-00500` in the `./results` directory.

The exact steps (starting at the root of the repository) are:

```sh
cd cleanup_scripts
mkdir -p wiki
cd wiki
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2    # Optionally use curl instead
bzip2 -d enwiki-latest-pages-articles-multistream.xml.bz2
cd ..    # back to cleanup_scripts
python3 WikiExtractor.py wiki/enwiki-latest-pages-articles-multistream.xml    # Results are placed in cleanup_scripts/text
./process_wiki.sh 'text/*/wiki_??'
```


## Credits

This data processing pipeline was adapted, with minimal changes, from the [reference implementation of the MLPerf BERT training benchmark](https://github.com/mlcommons/training/tree/8ca84826610f0becce960b80b2ad5cce8ddd66d8/language_model/tensorflow/bert/cleanup_scripts). It includes a copy of [WikiExtractor](https://github.com/attardi/wikiextractor/blob/e4abb4cbd019b0257824ee47c23dd163919b731b/WikiExtractor.py) pinned to a specific version known to work correctly.