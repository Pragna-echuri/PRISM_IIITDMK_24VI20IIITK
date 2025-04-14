import torch
import numpy as np
import imageio
from collections import OrderedDict
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from utils.util import instantiate_from_config
import os
from torch.utils.data import DataLoader
import argparse
import json
import cv2
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/mage+_caterv1.yaml')
# parser.add_argument('--bert-path', type=str, default='../bert-base-uncased/')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--checkpoint-path', type=str, default='../results//')

parser.add_argument('--device', type=str, default='cuda')
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:65532', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument("--n_samples", type=int, default=1, help="how many samples to produce for each instance",)
parser.add_argument("--test_model", type=str, default='models/MAGE+/catergenv2_diverse/model_best.pth')


# def convert_yuv_vedio_to_avi_vedio(input_folder, output_folder, width, height, fps=30):
#     if not os.path.exists(input_folder):
#         print(f"Error: Input folder {input_folder} does not exist")
#         return
    
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
    
#     yuv_files = [f for f in os.listdir(input_folder) if f.endswith('.yuv')]
#     total_files = len(yuv_files)
#     print(f"Found {total_files} YUV files to convert")
    
#     for idx, yuv_file in enumerate(yuv_files, 1):
#         input_path = os.path.join(input_folder, yuv_file)
#         output_file = os.path.splitext(yuv_file)[0] + '.avi'
#         output_path = os.path.join(output_folder, output_file)
        
#         print(f"[{idx}/{total_files}] Converting {yuv_file}...")
        
#         try:
#             y_size = width * height
#             uv_size = (width // 2) * (height // 2)
#             frame_size = y_size + 2 * uv_size
            
#             with open(input_path, 'rb') as f:
#                 fourcc = cv2.VideoWriter_fourcc(*'XVID')
#                 out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
#                 frame_count = 0
#                 while True:
#                     yuv_data = f.read(frame_size)
#                     if len(yuv_data) < frame_size:
#                         break
#                     y = np.frombuffer(yuv_data[:y_size], dtype=np.uint8).reshape((height, width))
#                     u = np.frombuffer(yuv_data[y_size:y_size + uv_size], dtype=np.uint8).reshape((height // 2, width // 2))
#                     v = np.frombuffer(yuv_data[y_size + uv_size:], dtype=np.uint8).reshape((height // 2, width // 2))
#                     u_full = cv2.resize(u, (width, height), interpolation=cv2.INTER_NEAREST)
#                     v_full = cv2.resize(v, (width, height), interpolation=cv2.INTER_NEAREST)
#                     yuv_frame = np.stack((y, u_full, v_full), axis=2)
#                     bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR)
#                     out.write(bgr_frame)
#                     frame_count += 1
#                     if frame_count % 100 == 0:
#                         print(f"Processed {frame_count} frames for {yuv_file}...")
                
#                 out.release()
#                 print(f"Completed: {output_file}")
        
#         except Exception as e:
#             print(f"Error converting {yuv_file}: {str(e)}")
#             if 'out' in locals():
#                 out.release()

import subprocess
import json
import sys
from pathlib import Path

def get_video_params (avi_path):
    """Extracts resolution, frame rate, pixel format, codec, and bit rate from AVI file."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,pix_fmt,bit_rate",
        "-of", "json",
        avi_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = json.loads(result.stdout)

    stream = info["streams"][0]
    codec = stream.get("codec_name")
    width = stream["width"]
    height = stream["height"]
    pix_fmt = stream["pix_fmt"]
    framerate = eval(stream["r_frame_rate"])  # e.g., "30/1" → 30.0
    bitrate = int(stream.get("bit_rate", 0))

    return {
        "codec": codec,
        "width": width,
        "height": height,
        "pix_fmt": pix_fmt,
        "framerate": framerate,
        "bitrate": bitrate
    }

def convert_yuv_vedio_to_avi_vedio(yuv_path, avi_template_path, output_avi_path):
    params = get_video_params(avi_template_path)

    cmd = [
        "ffmpeg",
        "-f", "rawvideo",
        "-pix_fmt", params["pix_fmt"],
        "-s", f"{params['width']}x{params['height']}",
        "-r", str(params["framerate"]),
        "-i", yuv_path,
        "-c:v", params["codec"]
    ]

    # Optional: Set bitrate if available
    if params["bitrate"] > 0:
        cmd += ["-b:v", str(params["bitrate"])]

    cmd.append(output_avi_path)

    print("Running command:")
    print(" ".join(cmd))

    subprocess.run(cmd)

    

def convert_yuv_to_avi():


    # Load the original JSON file
    input_file = "yuv_json.json"  # Change this to your actual filename
    output_dir = os.path.join("input_avi")  # Ensure this folder exists
    output_file = os.path.join(output_dir, "test_ambiguous.json")  # Full path

    # Load the original JSON file
    with open(input_file, "r") as f:
        data = json.load(f)

    # Modify the "video" values
    for key in data:
        old_video_path = data[key]["video"]
        new_path = old_video_path.replace("yuv_vedios", "videos").replace(".yuv", ".avi") # Modify as needed
        data[key]["video"] = new_path

    # Save the modified data to folder1
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Modified JSON saved as {output_file}")

    yuv_input = "CATER_GEN_v1_00010.avi"
    avi_reference = "CATER_GEN_v1_00010.avi"
    avi_output = "input_avi\\videos\CATER_GEN_v1_00007.avi"

    convert_yuv_vedio_to_avi_vedio(yuv_input, avi_reference, avi_output)


def sampling(opt):
    test_model = opt.test_model
    configs = OmegaConf.load(os.path.join(os.path.dirname(test_model), "config.yaml"))
    test_dataset = instantiate_from_config(configs.data, {'split': 'test'})
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    model = instantiate_from_config(configs.model)
    model = model.to(opt.device)

    if os.path.isfile(test_model):
        if opt.gpu is None:
            checkpoint = torch.load(test_model)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(opt.gpu)
            checkpoint = torch.load(test_model, map_location=loc)
            if list(checkpoint["state_dict"].keys())[0].startswith('module.'):
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(test_model))
    else:
        print("=> no checkpoint found at '{}'".format(test_model))

    model.eval()
    with torch.no_grad():
        idx = 0
        for batch in test_dataloader:
            if 'video_id' in batch.keys():
                video_id = batch['video_id'][0]
                del batch['video_id']
            for k in batch.keys():
                batch[k] = batch[k].to(opt.device)

            for iiidx in range(opt.n_samples):
                generated = model.autoregressive_generate(batch)
                generated.clamp_(min=-1, max=1)

            save_name = video_id + '-' + '{:.4f}'.format(batch['speed'][0].cpu().numpy())
            save_gifs(generated[0].cpu(), save_name, test_model)

            print(idx)
            idx += 1

def save_gifs(tgr, video_id, test_model):
    import imageio
    tgr_imgs = (tgr + 1) * 0.5
    tgr_imgs = (tgr_imgs * 255.).numpy().astype(np.uint8).transpose(0, 2, 3, 1)
    save_path = os.path.join(os.path.dirname("utils"), 'videos')
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    imageio.mimsave(os.path.join(save_path, video_id+'.gif'), tgr_imgs, fps=3)

if __name__=="__main__":
    convert_yuv_to_avi()
    opt = parser.parse_args()
    sampling(opt)
    

