PROJECT_DIR=$(git rev-parse --show-toplevel)
ENCODER_PATH=google-bert/bert-base-uncased
ENCODER_NAME=$(basename $ENCODER_PATH)
DEVICE=0

# Get embeddings
CUDA_VISIBLE_DEVICES=$DEVICE \
  CUDA_LAUNCH_BLOCKING=1 \
  python $PROJECT_DIR/src/faisslib/build_retriever.py \
    --data_dir $PROJECT_DIR/corpus/formatted/wikipedia-split \
    --model_path $ENCODER_PATH \
    --output_dir $PROJECT_DIR/metadata/wikipedia-split-$ENCODER_NAME \
    --device_id 0 \
    --do_chunk \
    --do_encode \
    --num_chunks_per_file 100000 \
    --batch_size 128 \

echo 'Build DB and index based on 60M wikipedia data'
CUDA_VISIBLE_DEVICES=$DEVICE \
  CUDA_LAUNCH_BLOCKING=1 \
  python $PROJECT_DIR/src/faisslib/build_retriever.py \
    --data_dir $PROJECT_DIR/corpus/formatted/wikipedia-split \
    --model_path $ENCODER_PATH \
    --output_dir $PROJECT_DIR/metadata/wikipedia-split-$ENCODER_NAME \
    --device_id -1 \
    --build_db \
    --build_index \
    --build_size 20000000 \
    --max_train_size 2000000 \
    --index_type IVF65536_HNSW32,PQ64 \
    --metric_type L2 \
    --sub_index_size 10000 \

echo 'Build DB and index based on all wikipedia data'
CUDA_VISIBLE_DEVICES=$DEVICE \
  CUDA_LAUNCH_BLOCKING=1 \
  python $PROJECT_DIR/src/faisslib/build_retriever.py \
    --data_dir $PROJECT_DIR/corpus/formatted/wikipedia-split \
    --model_path $ENCODER_PATH \
    --output_dir $PROJECT_DIR/metadata/wikipedia-split-$ENCODER_NAME \
    --device_id -1 \
    --build_db \
    --build_index \
    --build_size 40000000 \
    --max_train_size 2000000 \
    --index_type IVF65536_HNSW32,PQ64 \
    --metric_type L2 \
    --sub_index_size 10000 \
