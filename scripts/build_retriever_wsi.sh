PROJECT_DIR=$(git rev-parse --show-toplevel)
DEVICE=1

echo 'Prepare embeddings'
CUDA_VISIBLE_DEVICES=$DEVICE \
  python $PROJECT_DIR/src/preprocess/format_wsi.py \
    --input_dir /home/shared_files/FEATRUES/SELECTED \
    --output_dir $PROJECT_DIR/metadata/wsi

echo 'Build DB and index based on wsi data'
CUDA_VISIBLE_DEVICES=$DEVICE \
  python $PROJECT_DIR/src/faisslib/build_retriever.py \
    --output_dir $PROJECT_DIR/metadata/wsi \
    --device_id -1 \
    --build_db \
    --build_index \
    --train_ratio 0.2 \
    --least_num_train 1000 \
    --index_type IVF150_HNSW32,PQ64 \
    --metric_type L2 \
    --sub_index_size 10000
