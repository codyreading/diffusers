#!/bin/bash

MODEL_NAME="runwayml/stable-diffusion-v1-5"
OUTPUT_DIR="/localhome/cra80/Checkpoints/custom_diffusion"
FINE_TUNE_DIR=$PWD/fine-tuning
TOKEN="<new1>+<new2>"

accelerate launch examples/custom_diffusion/train_custom_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --concepts_list=$FINE_TUNE_DIR/concept_list.json \
  --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
  --resolution=512  \
  --train_batch_size=2  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --num_class_images=200 \
  --scale_lr --hflip  \
  --enable_xformers_memory_efficient_attention \
  --modifier_token $TOKEN \
  --report_to="wandb" \
  --resume_from_checkpoint latest \
  --no_safe_serialization \
  --push_to_hub
