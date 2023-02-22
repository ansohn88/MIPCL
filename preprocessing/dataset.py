import functools
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from utils import generate_transforms, return_cell_value, return_df_from_csv


class CytoBagDataset(Dataset):
    def __init__(self,
                 filename,
                 bags_dir,
                 class_path,
                 z_stack,
                 z_level=7,
                 pretrained=True,
                 ):
        self.pretrained = pretrained
        self.bags_dir = bags_dir
        self.filename = filename
        self.z_stack = z_stack
        if z_stack == False:
            self.z_level = z_level
        self.transforms = generate_transforms(pretrained=pretrained)

        self.bag, self.length = self.collate_bag_levels()

        self.class_label = self.retrieve_label(class_path)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = self.bag[index][2]
        img = Image.fromarray(img)
        # for batching, needs to be a 4D tensor
        img = self.transforms(img).unsqueeze(0)

        tile_z_level = self.bag[index][0]
        tile_coords = self.bag[index][1]

        return {"X": img,
                "y": self.class_label,
                "z_lvl": tile_z_level,
                "coord": tile_coords}

    def collate_bag_levels(self):
        stacked_bag_lvl_coords_tiles = list()
        bag_len = 0
        if self.z_stack:
            for i in range(1, 11):
                lvl_dir = f'{self.bags_dir}/level_{i}/{self.filename}.h5'
                if Path(lvl_dir).exists():
                    with h5py.File(lvl_dir, 'r') as f:
                        tiles = np.array(f['tiles'], dtype=np.uint8)
                        coords = np.array(f['coords'], dtype=np.int64)
                        z_level = np.repeat(int(i), coords.shape[0])

                    zipped = zip(z_level, coords, tiles)
                    stacked_bag_lvl_coords_tiles.extend(list(zipped))
                    bag_len += int(coords.shape[0])
                else:
                    pass
        else:
            z = self.z_level
            fp = f'{self.bags_dir}/level_{z}/{self.filename}.h5'
            with h5py.File(fp, 'r') as f:
                tiles = np.array(f['tiles'], dtype=np.uint8)
                coords = np.array(f['coords'], dtype=np.int64)
                z_level = np.repeat(z, coords.shape[0])

                zipped = zip(z_level, coords, tiles)
                stacked_bag_lvl_coords_tiles.extend(list(zipped))
                bag_len += int(coords.shape[0])

        return stacked_bag_lvl_coords_tiles, bag_len

    def retrieve_label(self, annotation_file):
        annotations = return_df_from_csv(annotation_file)
        og_fname = f'{self.filename[:-4]}.tif'
        diagnosis = return_cell_value(df=annotations,
                                      find_col='original_filename',
                                      find_value=og_fname,
                                      get_col='cyto_diag_groups')
        return int(diagnosis)
