import os
import cv2
import glob
import numpy as np
from tqdm import tqdm

# Save location for catagorical masks and current mask dir
cat_mask_save_dir = os.path.join('comma10k', 'cat_masks')
mask_dir = os.path.join('comma10k', 'masks')

m_split = '\\' if os.name == 'nt' else '/'

def mask2cat(mask_paths):
    print('Convering masks')
    for m in tqdm(mask_paths):
        # Read in images greyscale
        mask = cv2.imread(m, 0)
        unique_vales = np.unique(mask)
        # Only works for current unique values
        for cat_num, class_num in enumerate(unique_vales):
            mask[mask==class_num] = cat_num

        cv2.imwrite(os.path.join(cat_mask_save_dir, m.split(m_split)[-1]), mask)

if __name__ == "__main__":
    if not os.path.exists(cat_mask_save_dir):
        os.mkdir(cat_mask_save_dir)

    # Get all mask paths
    mask_file_paths = glob.glob(os.path.join(mask_dir, '*.png')
    mask2cat(mask_file_paths)
