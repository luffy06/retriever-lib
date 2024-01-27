DEVICE=1
ENCODER_PATH=/disk3/xy/LM/bert-base-uncased
PROJECT_DIR=/disk3/xy/PROJECT/wsy/retriever-lib
SPLIT=all

# Get embeddings
CUDA_VISIBLE_DEVICES=$DEVICE \
  CUDA_LAUNCH_BLOCKING=1 \
  python $PROJECT_DIR/src/faisslib/build_retriever.py \
    --data_dir $PROJECT_DIR/corpus/formatted/wikitext-103/$SPLIT \
    --model_path $ENCODER_PATH \
    --output_dir $PROJECT_DIR/metadata/wikitext-103-$SPLIT \
    --device_id 0 \
    --do_chunk \
    --do_encode \
    --num_chunks_per_file 1000000 \
    --batch_size 768 \

echo 'Build DB and index based on wikitext-103 '$SPLIT' data'
CUDA_VISIBLE_DEVICES=$DEVICE \
  CUDA_LAUNCH_BLOCKING=1 \
  python $PROJECT_DIR/src/faisslib/build_retriever.py \
    --data_dir $PROJECT_DIR/corpus/formatted/wikitext-103/$SPLIT \
    --model_path $ENCODER_PATH \
    --output_dir $PROJECT_DIR/metadata/wikitext-103-$SPLIT \
    --device_id 0 \
    --build_db \
    --build_index \
    --train_ratio 0.2 \
    --least_num_train 1000 \
    --index_type IVF65536_HNSW32,PQ64 \
    --metric_type L2 \
    --sub_index_size 10000 \
