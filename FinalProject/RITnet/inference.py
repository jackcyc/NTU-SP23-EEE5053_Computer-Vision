import argparse
import os
import random

import cv2
import numpy as np
import torch
from dataset import PupilDataset
from PIL import Image
from RITnet_v3 import DenseNet2D
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import get_predictions

# reproducibility
seed = 42
# Set the random seed for CPU
random.seed(seed)
# Set the random seed for NumPy
np.random.seed(seed)
# Set the random seed for GPU (if available)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
# Set the random seed for PyTorch
torch.manual_seed(seed)


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


def fit_ellipse(image_list: np.ndarray) -> torch.Tensor:
    '''
        Fit an ellipse to the maximal contour in the grayscale image.
        
        Parameters
        ----------
        image_list : np.ndarray (B, H, W). The grayscale image

        Returns
        -------
        refined_image_list : np.ndarray  (B, H, W). The refined image
    
    '''

    refined_image_list = np.zeros_like(image_list)
    for b in range(len(image_list)):
        image = image_list[b]

        # Find the contours in the binary image
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) >= 1:
            # Fit an ellipse to the maximal contour
            contour = max(contours, key=cv2.contourArea)
            try:
                ellipse = cv2.fitEllipse(contour)
            except:
                # print(f'cv2.fitEllipse(contour) failed')
                refined_image_list[b] = image
                continue

            image_refined = np.zeros_like(image)
            cv2.ellipse(image_refined, ellipse, 2, -1)

            refined_image_list[b] = image_refined
        else:
            refined_image_list[b] = image

    return refined_image_list.astype(np.uint8)


@torch.inference_mode()
def inference(model, testloader, device, output_dir):
    model.eval()

    pbar = tqdm(testloader)

    for i, batchdata in enumerate(pbar):
        img, labels, index, x, y = batchdata
        img = img.to(device)

        pred_map, _ = model(img)

        # # get raw probability map
        # pred_map = np.delete(pred_map.cpu().numpy(), 1, axis=1)
        # pred_map = torch.from_numpy(pred_map).softmax(dim=1)[:, 1, :, :].numpy()

        # get prediction
        pred_map = get_predictions(pred_map).numpy().astype(np.uint8)

        # Refine the prediction
        pred_map = fit_ellipse(pred_map)

        # save results for submission
        for j in range(len(index)):
            S, d, f = index[j].split('-')
            dir = os.path.join(output_dir, 'solution', S, d)
            os.makedirs(dir, exist_ok=True)

            Image.fromarray((pred_map[j] * 255).astype(np.uint8)).save(os.path.join(dir, f'{int(f)}.png'))


def main():
    args = parse_args()
    device = torch.device("cuda")

    # model
    model = DenseNet2D()
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint, strict=True)
    print(msg)
    model.to(device)
    model.eval()

    # data
    dataset = PupilDataset(root=args.data_dir, split='test')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

    inference(model, dataloader, device, args.output_dir)


if __name__ == '__main__':
    main()