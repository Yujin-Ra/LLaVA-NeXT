export CUDA_VISIBLE_DEVICES=4,5,6,7
MASTER_ADDR="127.0.0.1"
MASTER_PORT=16666


deepspeed --num_gpus=4 llava/mom/train.py \
  --model_path lmms-lab/LLaVA-Video-7B-Qwen2 \
  --video_dir llava/mom/dataset/NExTVideo/ \
  --train_ann llava/mom/dataset/train.json \
  --output_dir checkpoints/ \
  --deepspeed_config llava/mom/ds_config.json \
  --epochs 1 \
  --train_micro_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --save_steps 100 \
  --log_steps 50 \
  --master_port=$MASTER_PORT \
  --master_addr=$MASTER_ADDR