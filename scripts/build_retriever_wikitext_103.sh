PROJECT_DIR=$(git rev-parse --show-toplevel)
DEVICE=1
ENCODER_NAME=bert-large-uncased
ENCODER_PATH=$PROJECT_DIR/../models/$ENCODER_NAME
SPLIT=all

# Get embeddings
CUDA_VISIBLE_DEVICES=$DEVICE \
  python $PROJECT_DIR/src/faisslib/build_retriever.py \
    --data_dir $PROJECT_DIR/corpus/formatted/wikitext-103/$SPLIT \
    --model_path $ENCODER_PATH \
    --output_dir $PROJECT_DIR/metadata/wikitext-103-$SPLIT-$ENCODER_NAME \
    --device_id 0 \
    --do_chunk \
    --do_encode \
    --num_chunks_per_file 100000 \
    --batch_size 1024 \

echo 'Build DB and index based on wikitext-103 '$SPLIT' data'
CUDA_VISIBLE_DEVICES=$DEVICE \
  python $PROJECT_DIR/src/faisslib/build_retriever.py \
    --data_dir $PROJECT_DIR/corpus/formatted/wikitext-103/$SPLIT \
    --model_path $ENCODER_PATH \
    --output_dir $PROJECT_DIR/metadata/wikitext-103-$SPLIT-$ENCODER_NAME \
    --device_id -1 \
    --build_index \
    --train_ratio 0.2 \
    --least_num_train 1000 \
    --index_type IVF65536_HNSW32,PQ64 \
    --metric_type L2 \
    --sub_index_size 10000 \
