from torch.utils.data import Dataset
import os
import nibabel as nib
from ..processing.preprocess import znorm_rescale, center_crop
import numpy as np
import torch

class BratsDataset(Dataset):
    """Dataset class for loading BraTS training and test data.
    
    Args:
        data_dir: Directory of training or test data.
        mode: Either 'train' or 'test' specifying which data is being loaded.
    """
    def __init__(self, data_dir, mode, crop_size=None):
        self.data_dir = data_dir
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f'Dataset directory does not exist: {data_dir}')
        self.subject_list = sorted(
            subject for subject in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, subject))
        )
        if not self.subject_list:
            raise RuntimeError(f'No subject folders found in: {data_dir}')
        self.mode = mode
        self.crop_size = tuple(crop_size) if crop_size is not None else None

    def __len__(self):
        return len(self.subject_list)
    
    def load_nifti(self, subject_name, suffix):        
        """Loads nifti file for given subject and suffix."""

        nifti_filename = f'{subject_name}-{suffix}.nii.gz'
        nifti_path = os.path.join(self.data_dir, subject_name, nifti_filename)
        if not os.path.isfile(nifti_path):
            raise FileNotFoundError(
                f'Missing {suffix} image for subject {subject_name}: {nifti_path}'
            )
        nifti = nib.load(nifti_path)
        return nifti
    
    def load_subject_data(self, subject_name):
        """Loads images (and segmentation if in train mode) and extra info for a subject."""

        modalities_data = []
        for suffix in ['t1c', 't1n', 't2f', 't2w']:
            modality_nifti = self.load_nifti(subject_name, suffix)
            modality_data = modality_nifti.get_fdata()
            modalities_data.append(modality_data)

        if self.mode == 'train':
            seg_nifti = self.load_nifti(subject_name, 'seg')
            seg_data = seg_nifti.get_fdata()
            return modalities_data, seg_data
        elif self.mode == 'test':
            return modalities_data
    
    def __getitem__(self, idx):
        subject_name = self.subject_list[idx]

        # Load the data and extra info.
        if self.mode == 'train':
            imgs, seg = self.load_subject_data(subject_name)
        elif self.mode == 'test':
            imgs = self.load_subject_data(subject_name)

        # Do Z-score norm and rescaling preprocessing.
        imgs = [znorm_rescale(img) for img in imgs]

        # Perform center crop.
        imgs = [center_crop(img, self.crop_size) for img in imgs]

        imgs = [x[None, ...] for x in imgs]
        imgs = [np.ascontiguousarray(x, dtype=np.float32) for x in imgs]

        # Convert to torch tensors.
        imgs = [torch.from_numpy(x) for x in imgs]

        # If train mode, process segmentation similarly.
        if self.mode == 'train':
            # Use the identical deterministic spatial crop for every modality
            # and the segmentation.  No label information selects the crop.
            seg = center_crop(seg, self.crop_size)
            seg = seg[None, ...]
            seg = np.ascontiguousarray(seg)
            seg = torch.from_numpy(seg)

            return subject_name, imgs, seg
        
        elif self.mode == 'test':
            return subject_name, imgs
