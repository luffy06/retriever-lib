PROJECT_DIR=$(git rev-parse --show-toplevel)
DEVICE=0
ENCODER_PATH=google-bert/bert-base-uncased
ENCODER_NAME=$(basename $ENCODER_PATH)

# Get embeddings
CUDA_VISIBLE_DEVICES=$DEVICE \
  python $PROJECT_DIR/src/faisslib/build_retriever.py \
    --data_dir $PROJECT_DIR/corpus/formatted/fineweb \
    --model_path $ENCODER_PATH \
    --output_dir $PROJECT_DIR/metadata/fineweb-$ENCODER_NAME \
    --device_id 0 \
    --do_chunk \
    --do_encode \
    --num_chunks_per_file 100000 \
    --batch_size 256

if [ ! -f $PROJECT_DIR/metadata/fineweb-$ENCODER_NAME/metadata.json ]; then
  cp $PROJECT_DIR/metadata/fineweb-$ENCODER_NAME/metadata.json $PROJECT_DIR/metadata/fineweb-$ENCODER_NAME/metadata.json
fi

echo 'Build DB and index based on fineweb'
CUDA_VISIBLE_DEVICES=$DEVICE \
  python $PROJECT_DIR/src/faisslib/build_retriever.py \
    --data_dir $PROJECT_DIR/corpus/formatted/fineweb \
    --model_path $ENCODER_PATH \
    --output_dir $PROJECT_DIR/metadata/fineweb-$ENCODER_NAME \
    --device_id -1 \
    --build_db \
    --build_index \
    --train_ratio 0.2 \
    --least_num_train 1000 \
    --index_type IVF65536_HNSW32,PQ64 \
    --metric_type L2 \
    --sub_index_size 10000 \
    --use_embeddings_as_values
