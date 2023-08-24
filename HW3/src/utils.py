import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A = np.zeros((2 * N, 9))
    A[0::2, :] = np.hstack((u, np.ones((N, 1)), np.zeros((N, 3)), -u[:, 0:1] * v[:, 0:1], -u[:, 1:2] * v[:, 0:1], -v[:, 0:1])) 
    A[1::2, :] = np.hstack((np.zeros((N, 3)), u, np.ones((N, 1)), -u[:, 0:1] * v[:, 1:2], -u[:, 1:2] * v[:, 1:2], -v[:, 1:2]))

    # TODO: 2.solve H with A
    _, _, vh = np.linalg.svd(A)
    H = vh[-1, :].reshape(3, 3)

    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b', blending=False):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """
    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    mesh = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax)) 
    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    pixels = np.hstack((mesh[0].reshape(-1, 1), mesh[1].reshape(-1, 1), np.ones((mesh[0].size, 1)))) # x, y, 1

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        src_pixels = np.dot(H_inv, pixels.T).T
        src_pixels = src_pixels[:, :2] / src_pixels[:, 2:]
        src_pixels = np.round(src_pixels).astype(np.int)
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        mask = np.ones(src_pixels.shape[0], dtype=np.bool)
        mask = np.logical_and(mask, src_pixels[:, 0] >= 0)
        mask = np.logical_and(mask, src_pixels[:, 0] < w_src)
        mask = np.logical_and(mask, src_pixels[:, 1] >= 0)
        mask = np.logical_and(mask, src_pixels[:, 1] < h_src)
        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        src_pixels = src_pixels[mask]
        dst_pixels = np.round(np.stack((mesh[0], mesh[1]), axis=-1).reshape(-1, 2)[mask]).astype(np.int)

        # TODO: 6. assign to destination image with proper masking
        if not blending:
            dst[dst_pixels[:, 1], dst_pixels[:, 0]] = src[src_pixels[:, 1], src_pixels[:, 0]]

        # TODO: blending
        elif blending:
            src_cover = np.zeros((h_dst, w_dst))
            src_cover[dst_pixels[:, 1], dst_pixels[:, 0]] = 1
            dst_cover = np.zeros((h_dst, w_dst))
            dst_cover[np.sum(dst, axis=-1) != 0] = 1
            overlap = src_cover * dst_cover

            overlap_start = np.where(overlap.sum(axis=0)>5)[0][0]
            overlap_end = np.where(overlap.sum(axis=0)>5)[0][-1]+1

            dst_overlap = overlap.copy()
            src_overlap = overlap.copy()

            def sigmoid(x):
                return 1 / (1 + np.exp(-x))
            dst_overlap[:, overlap_start:overlap_end] *= sigmoid(np.linspace(-20, 20, num=overlap_end-overlap_start))
            src_overlap[:, overlap_start:overlap_end] *= sigmoid(np.linspace(20, -20, num=overlap_end-overlap_start))

            dst_weight = dst_cover - dst_overlap
            src_weight = src_cover - src_overlap

            warped_src = np.zeros_like(dst)
            warped_src[dst_pixels[:, 1], dst_pixels[:, 0]] = src[src_pixels[:, 1], src_pixels[:, 0]]

            dst = dst_weight[..., np.newaxis] * dst + src_weight[..., np.newaxis] * warped_src


    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        dst_pixels = np.dot(H, pixels.T).T
        dst_pixels = dst_pixels[:, :2] / dst_pixels[:, 2:]
        dst_pixels = np.round(dst_pixels).astype(np.int) # N, 2
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        mask = np.ones(dst_pixels.shape[0], dtype=np.bool)
        mask = np.logical_and(mask, dst_pixels[:, 0] >= 0)
        mask = np.logical_and(mask, dst_pixels[:, 0] < w_dst)
        mask = np.logical_and(mask, dst_pixels[:, 1] >= 0)
        mask = np.logical_and(mask, dst_pixels[:, 1] < h_dst)
        # TODO: 5.filter the valid coordinates using previous obtained mask
        src_pixels = np.round(np.stack((mesh[0], mesh[1]), axis=-1).reshape(-1, 2)[mask]).astype(np.int)
        dst_pixels = dst_pixels[mask]
        # TODO: 6. assign to destination image using advanced array indicing
        dst[dst_pixels[:, 1], dst_pixels[:, 0]] = src[src_pixels[:, 1], src_pixels[:, 0]]

    return dst
