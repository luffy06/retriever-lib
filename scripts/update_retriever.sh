DEVICE=0
PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
ENCODER_PATH=/root/autodl-tmp/wsy/models/gemma-2b
IFS='/' read -ra ADDR <<< "$ENCODER_PATH"
MODEL_NAME=${ADDR[-1]}
SOURCE_PATH=$PROJECT_DIR/metadata/wikitext-103-all-bert-large-uncased
TARGET_PATH=$PROJECT_DIR/metadata/wikitext-103-all-bert-large-uncased-$MODEL_NAME

CUDA_VISIBLE_DEVICES=$DEVICE \
  CUDA_LAUNCH_BLOCKING=1 \
  python $PROJECT_DIR/src/faisslib/update_retriever.py \
    --source_path $SOURCE_PATH \
    --target_path $TARGET_PATH \
    --model_name $ENCODER_PATH \
    --batch_size 4 \
    --max_length 8192 \
    --device $DEVICE
