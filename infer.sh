export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=$(dirname "$0"):$PYTHONPATH

/home/aix23103/anaconda3/envs/llava/bin/python \
    infer.py