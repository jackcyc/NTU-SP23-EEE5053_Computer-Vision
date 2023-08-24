import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    coverage = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    coverage[:imgs[0].shape[0], :imgs[0].shape[1]] = 1
    last_best_H = np.eye(3)
    out = None

    orb = cv2.ORB_create()
    bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        kp1, des1 = orb.detectAndCompute(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), None)
        kp2, des2 = orb.detectAndCompute(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY), None)

        matches = bfm.match(des1, des2)
        # matches = sorted(matches, key=lambda x: x.distance)[:len(matches)//4 ]

        dst_pt = np.array([kp1[m.queryIdx].pt for m in matches])
        src_pt = np.array([kp2[m.trainIdx].pt for m in matches])
        src_pt = np.concatenate((src_pt, np.ones((src_pt.shape[0], 1))), axis=1)

        # TODO: 2. apply RANSAC to choose best H
        iterations = 3000
        threshold = 5
        best_H = None
        best_inlier = 0
        for i in range(iterations):
            # sample 4 points
            random_matches = random.sample(matches, 4)
            rand_dst_pt = np.array([kp1[m.queryIdx].pt for m in random_matches])
            rand_src_pt = np.array([kp2[m.trainIdx].pt for m in random_matches])
            H = solve_homography(rand_src_pt, rand_dst_pt)
            
            # calculate inlier by vectorized method
            est_dst_pt = np.matmul(H, src_pt.T).T
            est_dst_pt = est_dst_pt[:, :2] / (est_dst_pt[:, 2:]+1e-7)
            inlier = np.sum(np.linalg.norm((dst_pt - est_dst_pt), axis=1) < threshold)
            
            # update best H
            if inlier >= best_inlier:
                best_inlier = inlier
                best_H = H
        
        # TODO: 3. chain the homographies
        last_best_H = np.matmul(last_best_H, best_H)

        # TODO: 4. apply warping
        dst = out = warping(im2, dst, last_best_H, 0, dst.shape[0], 0, dst.shape[1], 'b', blending=True)

    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)