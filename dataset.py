from typing import Tuple, Any

import numpy as np
from torchvision.datasets import ImageFolder


class ShipDataset(ImageFolder):
    def __init__(self, args, adaptive_aug: bool = False,  **kwargs):
        super(ShipDataset, self).__init__(args, **kwargs)
        self.should_transform = True
        self.adaptive_aug = adaptive_aug

    def update_iteration(self):
        if self.adaptive_aug:
            self.should_transform = not self.should_transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = sample.resize((256, 256))
        sample = np.asarray(sample).astype(np.float32)
        sample = np.moveaxis(sample, -1, 0)

        if self.transform is not None and self.should_transform:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
