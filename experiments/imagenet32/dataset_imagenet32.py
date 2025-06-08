import os
from torchvision import datasets
from torch.utils.data import Dataset

class ImageNet32Dataset(Dataset):
    """Simple ImageNet32 dataset wrapper using ImageFolder."""
    def __init__(self, root=None, split="train", transform=None):
        if root is None:
            root = os.environ.get("IMAGENET32_PATH", "./data/imagenet32")
        self.transform = transform
        split_dir = os.path.join(root, split)
        self.dataset = datasets.ImageFolder(split_dir, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return img
