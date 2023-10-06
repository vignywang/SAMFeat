from pathlib import Path
import argparse

import yaml
import numpy as np
from tqdm import tqdm

from hpatch_related.hpatch_dataset import OrgHPatchDataset
from models import get_model
import cv2 as cv
import torch
import time
def average_inference_time(time_collect):
    average_time = sum(time_collect) / len(time_collect)
    info = ('Average inference time: {}ms / {}fps'.format(
        round(average_time*1000), round(1/average_time))
    )
    print(info)
    return info

def extract_multiscale(net, img, scale_f=2 ** 0.25,
                       min_scale=0.125, max_scale=2.0,
                       min_size=0, max_size=9999,top_k=10000,
                       verbose=False):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    H, W,three= img.shape
    shape=img.shape
    assert three == 3, "should be a batch with a single RGB image"
    assert max_scale <= 2
    s = max_scale # current scale factor

    X, Y, S, C, Q, D = [], [], [], [], [], []
    while s + 0.001 >= max(min_scale, min_size / max(H, W)):
        if s - 0.001 <= min(max_scale, max_size / max(H, W)):
            nh = img.shape[0]
            nw = img.shape[1]
            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            # extract descriptors

            with torch.no_grad():
                res = net.predict(img=img)
            x = res['keypoints'][:,0]
            y = res['keypoints'][:,1]
            d = res['descriptors']
            scores = res['scores']

            X.append(x * W / nw)
            Y.append(y * H / nh)
            C.append(scores)
            D.append(d)

        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H * s), round(W * s)
        img = cv.resize(img, dsize=(nw, nh), interpolation=cv.INTER_LINEAR)
    torch.backends.cudnn.benchmark = old_bm
    Y = np.hstack(Y)
    X = np.hstack(X)
    scores = np.hstack(C)
    XY = np.stack([X, Y])
    XY = np.swapaxes(XY, 0, 1)
    D = np.vstack(D)
    idxs = scores.argsort()[-top_k or None:]
    predictions = {
        "keypoints": XY[idxs],
        "descriptors": D[idxs],
        "scores": scores[idxs],
        "shape": shape
    }

    return predictions
def extract_singlescale(net, img,top_k=10000 ,image_name=None):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    shape = img.shape
    X, Y, S, C, Q, D = [], [], [], [], [], []
    with torch.no_grad():
       #res = net.predict(img=img,image_name=image_name)
       res = net.predict(img=img)
    x = res['keypoints'][:,0]
    y = res['keypoints'][:,1]
    d = res['descriptors']
    scores = res['scores']

    X.append(x)
    Y.append(y)
    C.append(scores)
    D.append(d)
    torch.backends.cudnn.benchmark = old_bm
    Y = np.hstack(Y)
    X = np.hstack(X)
    scores = np.hstack(C)
    XY = np.stack([X, Y])
    XY = np.swapaxes(XY, 0, 1)
    D = np.vstack(D)
    idxs = scores.argsort()[-top_k or None:]
    predictions = {
        "keypoints": XY[idxs],
        "descriptors": D[idxs],
        "scores": scores[idxs],
        "shape": shape
    }

    return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,default='../configs/MTLDesc_eva.yaml')
    parser.add_argument('--single', type=bool, default=True)
    parser.add_argument('--output_root', type=str,default='hpatches_sequences/hpatches-sequences-release')
    parser.add_argument("--top-k", type=int, default=10000)
    parser.add_argument("--scale-f", type=float, default=2 ** 0.25)
    parser.add_argument("--min-size", type=int, default=0)
    parser.add_argument("--max-size", type=int, default=9999)
    parser.add_argument("--min-scale", type=float, default=0.3)
    parser.add_argument("--max-scale", type=float, default=1)
    parser.add_argument('--tag', type=str, default='mtldesc',required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    keys = '*' if config['keys'] == '*' else config['keys'].split(',')

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = OrgHPatchDataset(**config['hpatches'])
 #   time_collect = []
    with get_model(config['model']['name'])(**config['model']) as net:
        for i, data in tqdm(enumerate(dataset)):
            image_name = data['image_name']
            folder_name = data['folder_name']
            if args.single==True:
                start_time = time.time()
                predictions = extract_singlescale(net, data['image'],top_k=args.top_k,image_name=folder_name+'_'+image_name)
              #  time_collect.append(time.time() - start_time)
            else:
                predictions = extract_multiscale(net, data['image'], scale_f=args.scale_f,
                           min_scale=args.min_scale, max_scale=args.max_scale,
                           min_size=args.min_size, max_size=args.max_size,top_k=args.top_k,verbose=True)
            predictions['img_shape'] = data['image'].shape
            if config['output_type']=='benchmark':
                output_dir = Path(output_root,args.tag,folder_name)
                output_dir.mkdir(parents=True, exist_ok=True)
                outpath = Path(output_dir, image_name)
                np.savez(str(outpath), **predictions)
            else:
                output_dir = Path(output_root, folder_name)
                output_dir.mkdir(parents=True, exist_ok=True)
                outpath = Path(output_dir, image_name + '.ppm.' + args.tag)
                np.savez(open(outpath, 'wb'), **predictions)


