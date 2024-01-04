DEVICE=3
PROJECT_DIR=/disk3/xy/PROJECT/wsy/RetrieverLib

# CUDA_VISIBLE_DEVICES=$DEVICE \
#   CUDA_LAUNCH_BLOCKING=1 \
#   python $PROJECT_DIR/src/retriever/build_retriever.py \
#     --data_dir $PROJECT_DIR/corpus/formatted/wikitext-103 \
#     --model_path /disk3/xy/LM/bert-base-uncased \
#     --output_dir $PROJECT_DIR/metadata/wikitext103-bert-base \
#     --batch_size 32 \
#     --num_sentences_per_file 1000000 \
#     --device_id 0 \
#     --train_ratio 0.2 \
#     --least_num_train 1000 \
#     --index_type IVF65536_HNSW32,PQ64 \
#     --metric_type L2 \
#     --sub_index_size 10000 \
#     --verbose

CUDA_VISIBLE_DEVICES=$DEVICE \
  CUDA_LAUNCH_BLOCKING=1 \
  python $PROJECT_DIR/src/faisslib/build_retriever.py \
    --data_dir $PROJECT_DIR/corpus/formatted/wikipedia-split \
    --model_path /disk3/xy/LM/bert-base-uncased \
    --output_dir $PROJECT_DIR/metadata/wikipedia-split \
    --batch_size 32 \
    --num_sentences_per_file 1000000 \
    --device_id 0 \
    --train_ratio 0.2 \
    --least_num_train 1000 \
    --index_type IVF262144_HNSW32,PQ64 \
    --metric_type L2 \
    --sub_index_size 10000 \
    --verbose