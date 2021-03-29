from albumentations import (
    CLAHE, RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion,
    GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur,
    MedianBlur, IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast, RandomBrightness,
    HorizontalFlip, OneOf, Compose, Normalize
)
import torch
import numpy as np
import cv2

def strong_aug(p=0.5):
    return Compose([
        HorizontalFlip(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            always_apply=True)
    ], p=p)

class MaskImDataLoader:
    def __init__(self, image_paths, mask_paths, resize=None, mode='train'):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.resize = resize
        if mode == 'train':
            self.aug = strong_aug()
        else:
            self.aug = Compose([Normalize(always_apply=True)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = cv2.imread(self.image_paths[item])
        mask = cv2.imread(self.mask_paths[item], cv2.IMREAD_GRAYSCALE)

        if self.resize is not None:
            image = cv2.resize(image, self.resize)
            mask = cv2.resize(mask, self.resize)

        # add augmentation to image
        augmented = self.aug(image=image, mask=mask)
        image = augmented["image"]
        image = np.transpose(image, (2, 0, 1))
        mask = augmented["mask"]

        # Convert mask to catagorical
        mask = torch.from_numpy(mask).type(torch.long)
        # if one-hot needed uncomment
        #mask.unsqueeze_(0)
        #mask_onehot = torch.LongTensor(5, mask.size(1), mask.size(2))
        #mask_onehot.zero_()
        #mask_onehot.scatter_(0, mask.data, 1)

        return {"image": torch.tensor(image, dtype=torch.float),
                "mask": mask}
