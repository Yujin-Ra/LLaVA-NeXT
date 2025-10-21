export CUDA_VISIBLE_DEVICES=6

/home/aix23103/anaconda3/envs/llava/bin/python \
    llava/mom/infer.py \
    --model-path lmms-lab/LLaVA-Video-7B-Qwen2 \
    --video-dir llava/mom/dataset/NExTVideo/ \
    --dataset llava/mom/dataset/val.json \
    --result-path llava/mom/results/gridresults.json \