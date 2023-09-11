import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

# largely from RAFT
DEVICE = 'cuda'
RESOLUTION = (1280,720) 
ITERATION_N = 20
USE_MASK = False

def load_img(img_file):
    # img = Image.open(img_file).resize(RESOLUTION, Image.ANTIALIAS) # deprecated
    img = Image.open(img_file).resize(RESOLUTION, Image.LANCZOS)
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2,0,1).float()
    return img[None].to(DEVICE)

def load_masked_img(frame_path, mask_path):
    frame = Image.open(frame_path).resize(RESOLUTION, Image.LANCZOS)
    mask = Image.open(mask_path).resize(RESOLUTION, Image.LANCZOS).convert('RGB')
    frame = np.array(frame).astype(np.uint8)
    mask = np.array(mask).astype(np.uint8)
    mask = cv2.bitwise_not(mask)
    res = cv2.bitwise_or(frame, mask)
    res = torch.from_numpy(res).permute(2,0,1).float()
    return res[None].to(DEVICE) 


def inference(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = sorted(glob.glob(os.path.join(args.path_in, '*.jpg')))

        for img_path1, img_path2 in zip(images[:-1], images[1:]):
            if not USE_MASK:
                img1 = load_img(img_path1)
                img2 = load_img(img_path2)
            else:
                mask_path1 = args.mask + '\\' + os.path.basename(img_path1)[:-4] + '.png'
                mask_path2 = args.mask + '\\' + os.path.basename(img_path2)[:-4] + '.png'
                assert os.path.exists(mask_path1)
                assert os.path.exists(mask_path2)
                img1 = load_masked_img(img_path1, mask_path1)
                img2 = load_masked_img(img_path2, mask_path2)
            _, flow_up = model(img1, img2, iters=ITERATION_N, test_mode=True) # test_mode=True
            flow = flow_up[0].permute(1,2,0).cpu().numpy()
            flow = flow_viz.flow_to_image(flow)
            if not os.path.exists(args.path_out):
                os.mkdir(args.path_out)
            cv2.imwrite(args.path_out + '\\' + os.path.basename(img_path1)[:-4] + '.png', flow[:,:,[2,1,0]]) # no compression

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='checkpoint path')
    parser.add_argument('--path_in', help='frame directory path')
    parser.add_argument('--path_out', help='flow directory path')
    parser.add_argument('--mask', help='mask directory path')

    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    inference(args)

