from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
warnings.filterwarnings("ignore")

from llava.motion.processor import MotionFeatureExtractor, MotionVectorExtractor, MotionVectorProcessor, ResidualProcessor
from llava.motion.encoder import MVResidualModel, EarlyFusionProjector, LateFusionProjector

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()

video_path = "cat_and_chicken.mp4"

mve = MotionVectorExtractor(max_frames=12, sample_size=6)
mfe = MotionFeatureExtractor(width=224, height=224, frame_num=12)

frames, motions, motion_indices = mve(video_path)
features = mfe(frames, motions, motion_indices)
frame_time = list(range(0, len(frames)+1, 2))
frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

video = image_processor.preprocess(features["rgb"], return_tensors="pt")["pixel_values"]
batch_num = features['motion'].shape[0]
mvres = MVResidualModel(
        in_chans_mv=2,
        in_chans_res=3,
        dim=256,
        num_gops=batch_num,
        num_frames=12,
        depth=2,
        heads=4
    )
gop_ids = torch.tensor(list(range(batch_num)))
out = mvres(features['motion'], features['residual'], gop_ids)
fusion_layer = LateFusionProjector()




video = [video]
conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
time_instruciton = f"The video lasts for {len(frames)/2:.2f} seconds, and {len(frames)} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\nPlease describe this video in detail."
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()
input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
cont = model.generate(
    input_ids,
    images=video,
    modalities= ["video"],
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
print(text_outputs)