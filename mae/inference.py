import argparse
import os

import numpy as np
import torch
from models_vit import vit_large_patch16
from torch.utils.data import DataLoader
from tqdm import tqdm
from util.datasets import build_dataset_inference


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    args = parser.parse_args()

    print('Arguments:')
    for arg in vars(args):
        print(f'\t{arg}: {getattr(args, arg)}')
    return args


@torch.inference_mode()
def inference(model, dataloader):
    model.eval()
    preds = []
    for img in tqdm(dataloader):
        img = img.to('cuda', non_blocking=True)

        with torch.autocast(device_type='cuda', enabled=False):
            pred = model(img)
        pred = torch.softmax(pred, dim=1)[:, 1].cpu().numpy()

        preds.append(pred)

    preds = np.concatenate(preds, axis=0)

    return preds


def main():
    args = parse_args()
    data_root = args.data_dir
    output_dir = args.output_dir
    ckpt_path = args.ckpt_path

    # dataset
    dataset = build_dataset_inference(data_root)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

    # model
    model = vit_large_patch16(num_classes=2, global_pool=True)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=True)
    print(msg)
    model.to('cuda')

    # inference
    confs = inference(model, dataloader)

    # save raw confs
    cnts = dataset.cnts
    metadata = dataset.metadata

    subdirs = [os.path.join(output_dir, dir) for dir in metadata]

    assert cnts[-1] == len(confs), 'cnts[-1] != len(confs)'
    for i, (start, end) in enumerate(zip(cnts[:-1], cnts[1:])):
        confs_to_save = confs[start:end]
        os.makedirs(subdirs[i], exist_ok=True)
        np.savetxt(os.path.join(subdirs[i], 'conf.txt'), confs_to_save, fmt='%.4f', delimiter='\n')


main()
