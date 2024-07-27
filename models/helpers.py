import pandas as pd
import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset


# Parameters
NUM_CLASSES = 19
ORG_HEIGHT  = 1512
ORG_WIDTH   = 2016
# id_labels = {i: e for i, e in enumerate(METAINFO['classes'])}

METAINFO = {
    "classes": (
        "unlabelled",
        "asphalt/concrete",
        "dirt",
        "mud",
        "water",
        "gravel",
        "other-terrain",
        "tree-trunk",
        "tree-foliage",
        "bush/shrub",
        "fence",
        "other-structure",
        "pole",
        "vehicle",
        "rock",
        "log",
        "other-object",
        "sky",
        "grass",
    ),
    "palette": [
        (0  , 0  , 0  ),
        (230, 25 , 75 ),
        (60 , 180, 75 ),
        (255, 225, 25 ),
        (0  , 130, 200),
        (145, 30 , 180),
        (70 , 240, 240),
        (240, 50 , 230),
        (210, 245, 60 ),
        (250, 190, 190),
        (0  , 128, 128),
        (170, 110, 40 ),
        (255, 250, 200),
        (128, 0  , 0  ),
        (170, 255, 195),
        (128, 128, 0  ),
        (255, 215, 180),
        (0  , 0  , 128),
        (128, 128, 128),
    ],
    "cidx": list(range(NUM_CLASSES))
}


PATH = 'dataset/'
def get_dfs(path=PATH):
    train_df = pd.read_csv(PATH + 'train.csv')
    val_df   = pd.read_csv(PATH + 'val.csv'  )
    test_df  = pd.read_csv(PATH + 'test.csv' )
    return (train_df, val_df, test_df)


def label_to_rgb(label, palette):
    rgb_image = np.zeros((*label.shape, 3), dtype=np.uint8)
    for label_idx, color in enumerate(palette):
        rgb_image[label == label_idx] = color
    return rgb_image


class AugmentedSegmentationDataset(Dataset):
    def __init__(self, df, img_size_w, img_size_h, num_classes, transform=None):
        self.df = df
        self.img_size_w = img_size_w
        self.img_size_h = img_size_h
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['im_path']
        label_path = self.df.iloc[idx]['label_path']
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size_w, self.img_size_h))
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, (self.img_size_w, self.img_size_h))
        
        if self.transform:
            img = self.transform(img)
            label = torch.tensor(label, dtype=torch.long)
        return img, label


def load_asd(dfs, width, height, n_classes, transforms):
    datasets = [None] * len(dfs)
    for i, df in enumerate(dfs):
        dataset[i] = AugmentedSegmentationDataset(
            df, width, height, n_classes, transform=transforms[i]
        )
    return datasets