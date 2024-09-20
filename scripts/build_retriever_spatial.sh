PROJECT_DIR=$(git rev-parse --show-toplevel)
DEVICE=1

echo 'Prepare embeddings'
CUDA_VISIBLE_DEVICES=$DEVICE \
  python $PROJECT_DIR/src/preprocess/format_spatial.py \
    --input_dir $1 \
    --output_dir $PROJECT_DIR/metadata/gene

echo 'Build DB and index based on gene data'
CUDA_VISIBLE_DEVICES=$DEVICE \
  python $PROJECT_DIR/src/faisslib/build_retriever.py \
    --output_dir $PROJECT_DIR/metadata/gene \
    --device_id -1 \
    --build_db \
    --build_index \
    --train_ratio 1 \
    --least_num_train 1000 \
    --index_type IVF1200_HNSW32,PQ64 \
    --metric_type L2 \
    --sub_index_size 10000
