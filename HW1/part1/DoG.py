import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
        for oct in range(self.num_octaves):
            gaussian_images.append(image)
            for i in range(1, self.num_guassian_images_per_octave):
                gaussian_images.append(cv2.GaussianBlur(image, ksize=(0, 0), sigmaX = self.sigma**i))
            image = cv2.resize(gaussian_images[-1], (image.shape[1]//2, image.shape[0]//2), interpolation=cv2.INTER_NEAREST)
            

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for oct in range(self.num_octaves):
            for i in range(self.num_DoG_images_per_octave):
                second_img = gaussian_images[oct*self.num_guassian_images_per_octave + i + 1]
                first_img = gaussian_images[oct*self.num_guassian_images_per_octave + i]
                dog_images.append(cv2.subtract(second_img, first_img))

        self.dog_results = dog_images # for visualization

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        def find_local_extrema(imgs, threshold):
            keypoints = []
            B, H, W = imgs.shape
            for i in range(B-2):
                for x in range(W-2):
                    for y in range(H-2):
                        center = imgs[i+1, y+1, x+1]
                        if abs(center) >= threshold:
                            local_min = imgs[i:i+3, y:y+3, x:x+3].min()
                            local_max = imgs[i:i+3, y:y+3, x:x+3].max()
                            if (center == local_max) or (center == local_min):
                                keypoints.append([y+1, x+1])
            keypoints = np.array(keypoints, dtype=np.int32).reshape(-1, 2)
            return keypoints

        keypoints = []
        for oct in range(self.num_octaves):
            imgs = np.stack(dog_images[oct*self.num_DoG_images_per_octave : (oct+1)*self.num_DoG_images_per_octave], axis=0)
            keypoints.append(find_local_extrema(imgs, self.threshold) * 2**oct)
        keypoints = np.vstack(keypoints)

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints
