PROJECT_DIR=/disk3/xy/PROJECT/wsy/RetrieverLib
DATA_DIR=$PROJECT_DIR/corpus/original
OUTPUT_DIR=$PROJECT_DIR/corpus/formatted
CACHE_DIR=$DATA_DIR/.cache
PY_DIR=$PROJECT_DIR/src/preprocess

if [[ ! -d $CACHE_DIR ]]; then
  mkdir -p $CACHE_DIR
fi

if [[ ! -f $CACHE_DIR/wikitext-103-v1.zip ]]; then
  wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip -P $CACHE_DIR
fi

unzip $CACHE_DIR/wikitext-103-v1.zip -d $DATA_DIR
if [[ ! -d $OUTPUT_DIR/wikitext-103 ]]; then
  mkdir -p $OUTPUT_DIR/wikitext-103
fi
python $PY_DIR/format_wikitext.py --data_path $DATA_DIR/wikitext-103/wiki.train.tokens --output_path $OUTPUT_DIR/wikitext-103/wikitext103-train.json --raw_text
python $PY_DIR/format_wikitext.py --data_path $DATA_DIR/wikitext-103/wiki.valid.tokens --output_path $OUTPUT_DIR/wikitext-103/wikitext103-valid.json --raw_text
python $PY_DIR/format_wikitext.py --data_path $DATA_DIR/wikitext-103/wiki.test.tokens --output_path $OUTPUT_DIR/wikitext-103/wikitext103-test.json --raw_text

if [[ ! -f $CACHE_DIR/downloads/data/wikipedia_split/psgs_w100.tsv ]];then
  python $PROJECT_DIR/scripts/download_data.py --resource data.wikipedia_split.psgs_w100 --output_dir $CACHE_DIR
fi

if [[ ! -d $OUTPUT_DIR/wikipedia-split ]]; then
  mkdir -p $OUTPUT_DIR/wikipedia-split
fi
python $PY_DIR/format_wikipedia_split.py --data_path $CACHE_DIR/downloads/data/wikipedia_split/psgs_w100.tsv --output_path $OUTPUT_DIR/wikipedia-split/psgs_w100.json