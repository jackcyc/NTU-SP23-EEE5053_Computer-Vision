import numpy as np
import cv2.ximgproc as xip
import cv2


def computeDisp(Il, Ir, max_disp):
    # print(window_size, sigma_color, sigma_space)
    # np.random.seed(42)
    sigma_color=2
    sigma_space=7
    window_size=11
    Il_g = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY).astype(np.float32)
    Ir_g = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # myIl = np.concatenate((Il, np.expand_dims(Il_g, axis=-1)), axis=2)
    # myIr = np.concatenate((Ir, np.expand_dims(Ir_g, axis=-1)), axis=2)
    myIl = np.expand_dims(Il_g, axis=-1)
    myIr = np.expand_dims(Ir_g, axis=-1)
    # myIl = Il
    # myIr = Ir

    h, w, ch = myIl.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency

    # initialize 8-bit local binary pattern
    left_pattern = np.zeros((8, h, w, ch), dtype=bool) 
    right_pattern = np.zeros((8, h, w, ch), dtype=bool) 

    # compute local binary pattern
    # | 0 | 1 | 2 |
    # | 3 |   | 4 |
    # | 5 | 6 | 7 |
    idx = 0
    for y in range(-1, 2):
        for x in range(-1, 2):
            if x==0 and y==0:
                continue
            left_pattern[idx] = (myIl > np.roll(myIl, (x, y), axis=(1, 0)))
            right_pattern[idx] = (myIr > np.roll(myIr, (x, y), axis=(1, 0)))
            idx += 1
    
    # turn into (8*ch, h, w)
    left_pattern = np.transpose(left_pattern, (0, 3, 1, 2)).reshape((8*ch, h, w))
    right_pattern = np.transpose(right_pattern, (0, 3, 1, 2)).reshape((8*ch, h, w))

    # trim and pad border pattern - replicate
    left_pattern = left_pattern[:, 1:-1, 1:-1]
    right_pattern = right_pattern[:, 1:-1, 1:-1]
    left_pattern = np.pad(left_pattern, ((0, 0), (1, 1), (1, 1)), 'edge')
    right_pattern = np.pad(right_pattern, ((0, 0), (1, 1), (1, 1)), 'edge')

    # compute cost
    ltor_costs = np.zeros((max_disp+1, h, w), dtype=np.float32)
    rtol_costs = np.zeros((max_disp+1, h, w), dtype=np.float32)
    for disp in range(max_disp+1):
        cost = np.sum(np.logical_xor(left_pattern[:, :, disp:], right_pattern[:, :, :w-disp]), axis=0) # hamming distance
        ltor_costs[disp] = np.pad(cost, ((0, 0), (disp, 0)), 'edge')
        rtol_costs[disp] = np.pad(cost, ((0, 0), (0, disp)), 'edge')


    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    for disp in range(max_disp+1):
        ltor_costs[disp] = xip.jointBilateralFilter(Il_g, ltor_costs[disp], -1, sigma_color, sigma_space)
        rtol_costs[disp] = xip.jointBilateralFilter(Ir_g, rtol_costs[disp], -1, sigma_color, sigma_space)


    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    ltor_disparity = np.argmin(ltor_costs, axis=0)
    rtol_disparity = np.argmin(rtol_costs, axis=0)

    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    # 1. Left-right consistency check
    ltor_disparity = ltor_disparity.astype(np.float32)
    for y in range(h):
        for x in range(w):
            xx = round(x - ltor_disparity[y, x])
            if (xx < 0) or (ltor_disparity[y, x] != rtol_disparity[y, xx]):
                ltor_disparity[y, x] = np.nan

    # 2. Hole filling
    def ffill(arr):
        # https://stackoverflow.com/a/41191127/356463
        hole_mask = np.isnan(arr)
        idx = np.where(~hole_mask, np.arange(hole_mask.shape[1]),0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = arr[np.arange(idx.shape[0])[:,None], idx]
        return out
        
    ltor_disparity = np.pad(ltor_disparity, ((0, 0), (1, 1)), 'constant', constant_values=max_disp)
    
    F_L = ffill(ltor_disparity)
    F_R = ffill(ltor_disparity[:, ::-1])[:, ::-1]

    labels = np.min((F_L, F_R), axis=0)[:, 1:-1]

    # 3. Weighted median filtering
    labels = np.round(xip.weightedMedianFilter(Il.astype(np.uint8), labels.astype(np.uint8), window_size))

    # labels = ltor_disparity
    return labels.astype(np.uint8)