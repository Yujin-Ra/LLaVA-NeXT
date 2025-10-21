export CUDA_VISIBLE_DEVICES=7

accelerate launch --num_processes=1 \
-m lmms_eval \
--model llava_vid \
--model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average \
--tasks longvideobench_test_i \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_vid \
--output_path ./logs/