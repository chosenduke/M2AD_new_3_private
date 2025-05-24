from PIL import Image
import numpy as np
import cv2
import imgaug.augmenters as iaa
from .perlin import rand_perlin_2d_np
import glob
import torch
import os


class _Base_Anomaly_Generator():
    def __init__(self):
        pass

    def __call__(self, image:Image):

        ### just for examples -- should return a tuple of three elements
        augmented_image = image
        augmented_mask = np.zeros(image.shape[:2], dtype=np.float32)
        augmented_anomaly = 1

        return augmented_image, augmented_mask, augmented_anomaly

class WhiteNoiseGenerator(_Base_Anomaly_Generator):
    def __init__(self, max_try=200):
        super(WhiteNoiseGenerator, self).__init__()

        self.max_try = max_try

    def __call__(self, image:Image):
        processed_image = image.resize((1024, 1024))
        processed_image = cv2.cvtColor(np.asarray(processed_image), cv2.COLOR_RGB2BGR)
        processed_image = np.array(processed_image).astype(np.float32) / 255.0

        augmented_image, anomaly_mask, has_anomaly = self.augment_image_white_noise(processed_image)

        augmented_image = augmented_image * 255.0
        augmented_image = augmented_image.astype(np.uint8)

        augmented_image = Image.fromarray(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
        anomaly_mask = Image.fromarray(anomaly_mask[:, :, 0].astype(np.uint8) * 255, mode='L')

        return augmented_image, anomaly_mask, has_anomaly

    def augment_image_white_noise(self, image):

        # generate noise image
        noise_image = np.random.randint(0, 255, size=image.shape).astype(np.float32) / 255.0
        patch_mask = np.zeros(image.shape[:2], dtype=np.float32)

        # generate random mask
        patch_number = np.random.randint(0, 5)
        augmented_image = image

        for i in range(patch_number):
            try_count = 0
            coor_min_dim1 = 0
            coor_min_dim2 = 0

            coor_max_dim1 = 0
            coor_max_dim2 = 0
            while try_count < self.max_try:
                try_count += 1

                patch_dim1 = np.random.randint(image.shape[0] // 40, image.shape[0] // 10)
                patch_dim2 = np.random.randint(image.shape[1] // 40, image.shape[1] // 10)

                center_dim1 = np.random.randint(patch_dim1, image.shape[0] - patch_dim1)
                center_dim2 = np.random.randint(patch_dim2, image.shape[1] - patch_dim2)

                coor_min_dim1 = np.clip(center_dim1 - patch_dim1, 0, image.shape[0])
                coor_min_dim2 = np.clip(center_dim2 - patch_dim2, 0, image.shape[1])

                coor_max_dim1 = np.clip(center_dim1 + patch_dim1, 0, image.shape[0])
                coor_max_dim2 = np.clip(center_dim2 + patch_dim2, 0, image.shape[1])

                break

            patch_mask[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2] = 1.0

        augmented_image[patch_mask > 0] = noise_image[patch_mask > 0]

        patch_mask = patch_mask[:, :, np.newaxis]

        if patch_mask.max() > 0:
            has_anomaly = 1.0
        else:
            has_anomaly = 0.0

        return augmented_image, patch_mask, has_anomaly
        # return augmented_image, patch_mask, np.array([has_anomaly], dtype=np.float32)

class DRAEMGenerator(_Base_Anomaly_Generator):
    def __init__(self, anomaly_source_path):
        super(DRAEMGenerator, self).__init__()

        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path + "/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(image.shape[1], image.shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((image.shape[0], image.shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), 0.0
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1 - msk) * image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly = 0.0
            return augmented_image, msk, has_anomaly

    def transform_image(self, image_, anomaly_source_path):
        image = image_.copy()
        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        augmented_image = augmented_image * 255.0
        augmented_image = augmented_image.astype(np.uint8)
        augmented_image = Image.fromarray(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
        anomaly_mask = Image.fromarray(anomaly_mask[:, :, 0].astype(np.uint8) * 255, mode='L')
        return augmented_image, anomaly_mask, has_anomaly

    def __call__(self, image: Image):
        return self.transform_image(image_=np.asarray(image), anomaly_source_path=np.random.choice(self.anomaly_source_paths))