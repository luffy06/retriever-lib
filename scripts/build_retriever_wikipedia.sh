DEVICE=3
ENCODER_PATH=/disk3/xy/LM/bert-base-uncased
PROJECT_DIR=/disk3/xy/PROJECT/wsy/retriever-lib

# Get embeddings
CUDA_VISIBLE_DEVICES=$DEVICE \
  CUDA_LAUNCH_BLOCKING=1 \
  python $PROJECT_DIR/src/faisslib/build_retriever.py \
    --data_dir $PROJECT_DIR/corpus/formatted/wikipedia-split \
    --model_path $ENCODER_PATH \
    --output_dir $PROJECT_DIR/metadata/wikipedia-split \
    --device_id 0 \
    --do_chunk \
    --do_encode \
    --num_chunks_per_file 1000000 \
    --batch_size 256 \

echo 'Build DB and index based on 100K data'
CUDA_VISIBLE_DEVICES=$DEVICE \
  CUDA_LAUNCH_BLOCKING=1 \
  python $PROJECT_DIR/src/faisslib/build_retriever.py \
    --data_dir $PROJECT_DIR/corpus/formatted/wikipedia-split \
    --model_path $ENCODER_PATH \
    --output_dir $PROJECT_DIR/metadata/wikipedia-split \
    --device_id 0 \
    --build_size 100000 \
    --build_db \
    --build_index \
    --train_ratio 0.2 \
    --least_num_train 1000 \
    --index_type IVF1265_HNSW32,PQ64 \
    --metric_type L2 \
    --sub_index_size 10000 \

echo 'Build DB and index based on 500K data'
CUDA_VISIBLE_DEVICES=$DEVICE \
  CUDA_LAUNCH_BLOCKING=1 \
  python $PROJECT_DIR/src/faisslib/build_retriever.py \
    --data_dir $PROJECT_DIR/corpus/formatted/wikipedia-split \
    --model_path $ENCODER_PATH \
    --output_dir $PROJECT_DIR/metadata/wikipedia-split \
    --device_id 0 \
    --build_size 500000 \
    --build_db \
    --build_index \
    --train_ratio 0.2 \
    --least_num_train 1000 \
    --index_type IVF2828_HNSW32,PQ64 \
    --metric_type L2 \
    --sub_index_size 10000 \

echo 'Build DB and index based on 1M data'
CUDA_VISIBLE_DEVICES=$DEVICE \
  CUDA_LAUNCH_BLOCKING=1 \
  python $PROJECT_DIR/src/faisslib/build_retriever.py \
    --data_dir $PROJECT_DIR/corpus/formatted/wikipedia-split \
    --model_path $ENCODER_PATH \
    --output_dir $PROJECT_DIR/metadata/wikipedia-split \
    --device_id 0 \
    --build_size 1000000 \
    --build_db \
    --build_index \
    --train_ratio 0.2 \
    --least_num_train 1000 \
    --index_type IVF4000_HNSW32,PQ64 \
    --metric_type L2 \
    --sub_index_size 10000 \

echo 'Build DB and index based on 10M data'
CUDA_VISIBLE_DEVICES=$DEVICE \
  CUDA_LAUNCH_BLOCKING=1 \
  python $PROJECT_DIR/src/faisslib/build_retriever.py \
    --data_dir $PROJECT_DIR/corpus/formatted/wikipedia-split \
    --model_path $ENCODER_PATH \
    --output_dir $PROJECT_DIR/metadata/wikipedia-split \
    --device_id 0 \
    --build_size 10000000 \
    --build_db \
    --build_index \
    --train_ratio 0.2 \
    --least_num_train 1000 \
    --index_type IVF65536_HNSW32,PQ64 \
    --metric_type L2 \
    --sub_index_size 10000 \

echo 'Build DB and index based on 20M data'
CUDA_VISIBLE_DEVICES=$DEVICE \
  CUDA_LAUNCH_BLOCKING=1 \
  python $PROJECT_DIR/src/faisslib/build_retriever.py \
    --data_dir $PROJECT_DIR/corpus/formatted/wikipedia-split \
    --model_path $ENCODER_PATH \
    --output_dir $PROJECT_DIR/metadata/wikipedia-split \
    --device_id 0 \
    --build_size 20000000 \
    --build_db \
    --build_index \
    --train_ratio 0.2 \
    --least_num_train 1000 \
    --index_type IVF262144_HNSW32,PQ64 \
    --metric_type L2 \
    --sub_index_size 10000 \
