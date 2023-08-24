import numpy as np
import cv2


class Joint_bilateral_filter(object):

    def __init__(self, sigma_s, sigma_r):
        # self.sigma_s = sigma_s
        # self.sigma_r = sigma_r
        self.pad_w = int(3 * sigma_s)
        self.window_size = 2 * self.pad_w + 1

        # spatial kernel
        spatial_kernel_const = 1 / (2 * (sigma_s**2))
        mesh = np.stack(np.meshgrid(np.arange(-self.pad_w, self.pad_w + 1), np.arange(-self.pad_w, self.pad_w + 1)), axis=-1)
        self.w_s = np.exp(-np.sum(mesh**2, axis=-1) * spatial_kernel_const)
        # range kernel
        self.range_kernel_const = 1 / (2 * (sigma_r**2))

    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.float64)

        padded_guidance /= 255.
        if len(padded_guidance.shape) == 2:
            padded_guidance = padded_guidance[:, :, np.newaxis]

        output = np.zeros_like(img)

        H, W = img.shape[:2]
        for i_start in range(H):
            for j_start in range(W):
                i_end = i_start + self.window_size 
                j_end = j_start + self.window_size 

                # range kernel
                w_r = np.exp(-(np.square(padded_guidance[i_start+self.pad_w , j_start+self.pad_w] - padded_guidance[i_start:i_end, j_start:j_end])).sum(axis=-1) * self.range_kernel_const)

                weight = self.w_s * w_r
                output[i_start, j_start] = (weight[:, :, np.newaxis] * padded_img[i_start:i_end, j_start:j_end]).sum((0, 1)) / weight.sum()

        return np.clip(output, 0, 255).astype(np.uint8)