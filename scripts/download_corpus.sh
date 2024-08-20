PROJECT_DIR=$(git rev-parse --show-toplevel)
DATA_DIR=$PROJECT_DIR/corpus/original
OUTPUT_DIR=$PROJECT_DIR/corpus/formatted
PY_DIR=$PROJECT_DIR/src/preprocess

if [[ ! -d $DATA_DIR ]]; then
    mkdir -p $DATA_DIR
fi

if [[ ! -d $OUTPUT_DIR ]]; then
    mkdir -p $OUTPUT_DIR
fi

if [[ ! -d $DATA_DIR/wikitext ]]; then
    huggingface-cli download --repo-type dataset \
        --resume-download Salesforce/wikitext \
        --local-dir $DATA_DIR/wikitext
fi

echo 'Format wikitext-103 (train)'
if [[ ! -d $OUTPUT_DIR/wikitext-103/all ]]; then
    mkdir -p $OUTPUT_DIR/wikitext-103/all
fi
python $PY_DIR/format_wikitext.py \
    --data_path $DATA_DIR/wikitext/wikitext-103-raw-v1/train-00000-of-00002.parquet \
                $DATA_DIR/wikitext/wikitext-103-raw-v1/train-00001-of-00002.parquet \
    --output_path $OUTPUT_DIR/wikitext-103/all/wikitext103-train.json

echo 'Format wikitext-103 (valid)'
python $PY_DIR/format_wikitext.py \
    --data_path $DATA_DIR/wikitext/wikitext-103-raw-v1/validation-00000-of-00001.parquet \
    --output_path $OUTPUT_DIR/wikitext-103/all/wikitext103-valid.json

echo 'Format wikitext-103 (test)'
python $PY_DIR/format_wikitext.py \
    --data_path $DATA_DIR/wikitext/wikitext-103-raw-v1/test-00000-of-00001.parquet \
    --output_path $OUTPUT_DIR/wikitext-103/all/wikitext103-test.json

# This requries a lot of space, if you don't have enough space, you can skip this
# if [[ ! -f $CACHE_DIR/downloads/data/wikipedia_split/psgs_w100.tsv ]];then
#   python $PROJECT_DIR/scripts/download_data.py --resource data.wikipedia_split.psgs_w100 --output_dir $CACHE_DIR
# fi

# if [[ ! -d $OUTPUT_DIR/wikipedia-split ]]; then
#   mkdir -p $OUTPUT_DIR/wikipedia-split
# fi
# echo 'Format wikipedia-split'
# python $PY_DIR/format_wikipedia_split.py --data_path $CACHE_DIR/downloads/data/wikipedia_split/psgs_w100.tsv --output_path $OUTPUT_DIR/wikipedia-split/psgs_w100.json