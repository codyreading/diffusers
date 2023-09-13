#!/bin/bash

MODEL_NAME="runwayml/stable-diffusion-v1-5"
PROJ_DIR=$PWD
FINE_TUNE_DIR=$PWD/fine-tuning
INSTANCE_DIR=$FINE_TUNE_DIR/datasets/car
OUTPUT_DIR=$FINE_TUNE_DIR/checkpoints


CMD="accelerate launch examples/dreambooth/train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt='a photo of sks car' \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler='constant' \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt='A photo of sks car on the road' \
  --validation_epochs=50 \
  --seed="0" \
  --push_to_hub"

  echo $CMD
  eval $CMD