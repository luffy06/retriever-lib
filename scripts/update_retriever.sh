PROJECT_DIR=$(git rev-parse --show-toplevel)
ENCODER_PATH=google/gemma-2b
MODEL_NAME=$(basename $ENCODER_PATH)
SOURCE_PATH=$PROJECT_DIR/metadata/wikitext-103-all-bert-large-uncased
TARGET_PATH=$PROJECT_DIR/metadata/wikitext-103-all-bert-large-uncased-$MODEL_NAME
DEVICE=0

CUDA_VISIBLE_DEVICES=$DEVICE \
  CUDA_LAUNCH_BLOCKING=1 \
  python $PROJECT_DIR/src/faisslib/update_retriever.py \
    --source_path $SOURCE_PATH \
    --target_path $TARGET_PATH \
    --model_name $ENCODER_PATH \
    --batch_size 2 \
    --max_length 8192 \
    --device $DEVICE
