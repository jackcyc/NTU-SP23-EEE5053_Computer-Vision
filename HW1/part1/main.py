import numpy as np
import cv2
import argparse
from DoG import Difference_of_Gaussian
import os


def plot_keypoints(img_gray, keypoints, save_path):
    img = np.repeat(np.expand_dims(img_gray, axis = 2), 3, axis = 2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)

def main():
    ### TODO ###
    if not os.path.exists('./DoG_results'):
        os.mkdir('./DoG_results')
    
    # report 1
    image_path = './testdata/1.png'
    print(f'Processing {image_path} ...')
    img = cv2.imread(image_path, 0).astype(np.float32)
    DoG = Difference_of_Gaussian(threshold=3.0)
    keypoints = DoG.get_keypoints(img)

    for i in range(1, 3):
        for j in range(1, 5):
            img = DoG.dog_results[(i-1)*4+(j-1)]
            # minmax normalization
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            cv2.imwrite(f'./DoG_results/DoG{i}-{j}.png', img.astype(np.float32)*255)

    # report 2
    image_path = './testdata/2.png'
    print(f'Processing {image_path} ...')
    img = cv2.imread(image_path, 0).astype(np.float32)
    for threshold in range(1, 4):
        DoG = Difference_of_Gaussian(threshold=threshold)
        keypoints = DoG.get_keypoints(img)
        plot_keypoints(img, keypoints, f'./DoG_results/keypoints_th{threshold}.png')


if __name__ == '__main__':
    main()