import os
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description='Fit ellipse to binary images')
    parser.add_argument('--solution_path', '-o', type=str, default='../result/solution')
    args = parser.parse_args()

    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    return args

def find_closed_eye(image: np.ndarray):
    '''
        Find the closed eye in the binary image.
        Args:
            image: binary image
        Returns:
            pupil_size: mean value of the ellipse
            is_invalid: True if the pupil is on the margin or not valid
    '''
    h, w = image.shape

    # Find the contours in the binary image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 1:
        # Test 1: check if the image is empty or has more than one contour
        return 0.0, True
    else:
        # Test 2: check if the contour is too small
        try:
            center, axes, angle = cv2.fitEllipse(contours[0])
        except:
            return 0.0, True
        
        # Test 3: check if the ellipse is severly out of the image
        img1 = np.zeros_like(image)
        img2 = np.zeros_like(image)
        cv2.ellipse(img1, (int(center[0]), int(center[1])), (int(axes[0]), int(axes[1])), angle, 0, 360, 255, -1)
        cv2.ellipse(img2, (w //2, h // 2), (int(axes[0]), int(axes[1])), angle, 0, 360, 255, 1)
        if np.sum(img1) / np.sum(img2) < 0.8:
            return 0.0, True
        
        # Test 4: check if the ellipse is close to the margin
        if center[0] < 0.1 * w or center[0] > 0.9 * w or center[1] < 0.1 * h or center[1] > 0.9 * h:
            return np.mean(img1), True
        else:
            return np.mean(img1), False

def postprocess(solution_path: str):
    d_dir = sorted(glob.glob(os.path.join(solution_path, 'solution', 'S*', '*')),
                    key=lambda x: int(x.split('/')[-2][-1]) * 100 + int(x.split('/')[-1]))

    pbar = tqdm(d_dir)
    for d in pbar:
        pbar.set_description(f'Processing {d[-5:]}')
        file_list = sorted([name for name in os.listdir(d) if name.endswith('.png')],
                            key=lambda x: int(x.split('.')[0]))
        
        # Load confidence txt
        confidence = np.loadtxt(os.path.join(d, 'conf.txt'))

        # Post-processing: calculate size and margin info of each pupil
        sizes, invalids = [], []
        for f in file_list:
            img_path = os.path.join(d, f)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            size, invalid = find_closed_eye(img)
            sizes.append(size)
            invalids.append(invalid)

        # Post-processing1: set must-be-zero confidence to zero
        sizes = np.array(sizes, dtype=np.float32)
        invalids = np.array(invalids, dtype=np.int8)

        median_pupil_size = np.median(sizes[sizes > 0])
        sizes[sizes < 0.5 * median_pupil_size] = 0
        zero_conf_idx = np.where(np.logical_and(sizes == 0, invalids == 1))[0]
        confidence[zero_conf_idx] = 0

        # Post-processing2: clip confident prediction 
        confidence[confidence>0.75] = 1
        confidence[confidence<0.25] = 0

        np.savetxt(os.path.join(d, 'conf.txt'), confidence, fmt='%.4f')

if __name__ == '__main__':
    args = get_args()
    postprocess(args.solution_path)