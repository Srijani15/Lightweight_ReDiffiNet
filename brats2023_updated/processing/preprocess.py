"""This module contains functions for preprocessing MRI images and segmentations."""

import numpy as np
from skimage import exposure

def znorm_rescale(img):
    """Applies Z-score normalization and rescaling to a MRI image."""

    # Z-score norm
    movingNan=np.copy(img)
    movingNan[movingNan==0]=np.nan
    movingMean=np.nanmean(movingNan)
    movingSTD=np.nanstd(movingNan)
    moving=(img-movingMean)/movingSTD
    b=255/(1-(moving.max()/moving.min()))
    a=-b/moving.min()
    movingNorm=np.copy(moving)
    movingNorm=np.round((movingNorm*a)+b,2)

    # Rescaling
    p2, p98 = np.percentile(movingNorm, (1, 99)) # These parameters may not be optimal, further testing could be done
    moving_rescale = exposure.rescale_intensity(movingNorm, in_range=(p2, p98))

    return moving_rescale

# Historical crop ranges.  They are kept as the default so existing baseline
# experiments reproduce their preprocessing exactly.
X_START, X_END, Y_START, Y_END, Z_START, Z_END = (56,184, 24,216, 14,142)
DEFAULT_CROP_SIZE = (X_END - X_START, Y_END - Y_START, Z_END - Z_START)


def center_crop(img, crop_size=None):
    """Center crop an MRI image or segmentation with the same spatial slices.

    Passing ``None`` uses the project's original fixed ranges exactly.  A
    supplied ``(x, y, z)`` crop is deterministic and centred in the image,
    which makes reduced-ROI experiments explicit without using labels.
    """
    if crop_size is None:
        return img[X_START:X_END, Y_START:Y_END, Z_START:Z_END]

    crop_size = tuple(int(value) for value in crop_size)
    if len(crop_size) != 3 or any(value < 1 for value in crop_size):
        raise ValueError('crop_size must contain three positive integers.')
    if img.ndim < 3:
        raise ValueError(f'Expected at least three spatial dimensions, got {img.shape}.')
    image_shape = img.shape[:3]
    if any(crop > image for crop, image in zip(crop_size, image_shape)):
        raise ValueError(f'crop_size {crop_size} exceeds image shape {image_shape}.')

    starts = tuple((image - crop) // 2 for crop, image in zip(crop_size, image_shape))
    slices = tuple(slice(start, start + crop) for start, crop in zip(starts, crop_size))
    return img[slices]

def undo_center_crop(input):
    """Undos center crop of a MRI image (or seg)."""
    out = np.zeros((240, 240, 155))
    out[X_START:X_END, Y_START:Y_END, Z_START:Z_END] = input 
    return out
