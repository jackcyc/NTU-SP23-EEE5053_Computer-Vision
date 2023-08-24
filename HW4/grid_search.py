import numpy as np
import cv2
import argparse
import os, glob
from eval import evaluate
from computeDisp import computeDisp

from multiprocessing import Pool
import time



parser = argparse.ArgumentParser(description='main function of Stereo matching')
parser.add_argument('--dataset_path', default='./testdata/', help='path to dataset')
# parser.add_argument('--image', choices=['Tsukuba', 'Venus', 'Teddy', 'Cones'], required=True, help='choose processing image')
args = parser.parse_args()

config = {'Tsukuba': (15, 16),
            'Venus':   (20, 8),
            'Teddy':   (60, 4),
            'Cones':   (60, 4)}

img_lefts = []
img_rights = []
img_gts = []
max_disps = []
scale_factors = []

for i in range(4):  
    args.image = list(config.keys())[i]
    img_lefts.append(cv2.imread(os.path.join(args.dataset_path, args.image, 'img_left.png')))
    img_rights.append(cv2.imread(os.path.join(args.dataset_path, args.image, 'img_right.png')))
    gt_path = glob.glob(os.path.join(args.dataset_path, args.image, 'disp_gt.*'))[0]
    img_gts.append(cv2.imread(gt_path, -1))
    max_disp, scale_factor = config[args.image]
    max_disps.append(max_disp)
    scale_factors.append(scale_factor)


def eva(i, win, sig_c, sig_s):
    labels = computeDisp(img_lefts[i], img_rights[i], max_disps[i], window_size=win, sigma_color=sig_c, sigma_space=sig_s)
    error = evaluate(labels, img_gts[i], scale_factors[i])
    error = round(error, 5)

    return {f'{win}': error, f'{20-win}': 0}

pools = Pool(10)



min_err = 100
for sig_s in range(3, 11):
    for sig_c in range(1, 9):
        # for win in range(9, 13, 2):
            print('sig_c: ', sig_c, 'sig_s: ', sig_s)
            t0 = time.time()
            total_err = 0
            results = pools.starmap(eva, [(i, win, sig_c, sig_s) for i in range(4) for win in range(9, 13, 2)])


            win9 = 0
            win11 = 0
            for result in results:
                win9 += result['9']
                win11 += result['11']
            total_err = min(win9, win11)
            win = 9 if win9 < win11 else 11
            # for i in range(4):  
            #     args.image = list(config.keys())[i]

            #     labels = computeDisp(img_lefts[i], img_rights[i], max_disps[i], window_size=win)

            #     if glob.glob(os.path.join(args.dataset_path, args.image, 'disp_gt.*')):
            #         gt_path = glob.glob(os.path.join(args.dataset_path, args.image, 'disp_gt.*'))[0]
            #         img_gt = cv2.imread(gt_path, -1)
            #         error = evaluate(labels, img_gt, scale_factors[i])
            #         # print('[Bad Pixel Ratio] %.2f%%' % (error*100))
            #         total_err += round(error, 4)
            # print('time: ', time.time()-t0)
            print(f'{total_err*100:.3f}')
            if total_err < min_err:
                min_err = total_err
                best_win = win
                best_sig_c = sig_c
                best_sig_s = sig_s
                print('best window_size: ', best_win, 'best sig_c: ', best_sig_c, 'best sig_s: ', best_sig_s, 'min_err: ', min_err)
print('best window_size: ', best_win, 'best sig_c: ', best_sig_c, 'best sig_s: ', best_sig_s)
    

