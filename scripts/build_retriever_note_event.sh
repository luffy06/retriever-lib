DEVICE=0
ENCODER_NAME=bert-large-uncased
ENCODER_PATH=/mnt/wsy/models/$ENCODER_NAME
PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")

# Get embeddings
CUDA_VISIBLE_DEVICES=$DEVICE \
  CUDA_LAUNCH_BLOCKING=1 \
  python $PROJECT_DIR/src/faisslib/build_retriever.py \
    --data_dir $PROJECT_DIR/corpus/formatted/note-event \
    --model_path $ENCODER_PATH \
    --output_dir $PROJECT_DIR/metadata/note-event-$ENCODER_NAME \
    --device_id 0 \
    --do_chunk \
    --do_encode \
    --num_chunks_per_file 100000 \
    --batch_size 128 \

echo 'Build DB and index based on 20M note-event data'
CUDA_VISIBLE_DEVICES=$DEVICE \
  CUDA_LAUNCH_BLOCKING=1 \
  python $PROJECT_DIR/src/faisslib/build_retriever.py \
    --data_dir $PROJECT_DIR/corpus/formatted/note-event \
    --model_path $ENCODER_PATH \
    --output_dir $PROJECT_DIR/metadata/note-event-$ENCODER_NAME \
    --device_id -1 \
    --build_db \
    --build_index \
    --build_size 20000000 \
    --max_train_size 2000000 \
    --index_type IVF65536_HNSW32,PQ64 \
    --metric_type L2 \
    --sub_index_size 10000 \

echo 'Build DB and index based on all note-event data'
CUDA_VISIBLE_DEVICES=$DEVICE \
  CUDA_LAUNCH_BLOCKING=1 \
  python $PROJECT_DIR/src/faisslib/build_retriever.py \
    --data_dir $PROJECT_DIR/corpus/formatted/note-event \
    --model_path $ENCODER_PATH \
    --output_dir $PROJECT_DIR/metadata/note-event-$ENCODER_NAME \
    --device_id -1 \
    --build_db \
    --build_index \
    --max_train_size 2000000 \
    --index_type IVF65536_HNSW32,PQ64 \
    --metric_type L2 \
    --sub_index_size 10000 \
