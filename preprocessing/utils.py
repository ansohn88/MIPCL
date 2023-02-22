from pathlib import Path, PosixPath
from typing import List, Optional, Union

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from torchvision import transforms


def save_hdf5(
        output_path: str,
        asset_dict: dict,
        attr_dict: Optional[dict] = None,
        mode: str = 'a'
) -> str:
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(
                key,
                shape=data_shape,
                maxshape=maxshape,
                chunks=chunk_shape,
                dtype=data_type
            )
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path


def does_h5_exist(
        input_file: Union[PosixPath, str],
        output_dir: Union[PosixPath, str]
) -> bool:
    from pyvips import Image
    if type(outdir_parent) == str:
        outdir_parent = Path(output_dir)

    num_of_lvls = Image.new_from_file(str(input_file)).get('n-pages')
    num_of_lvls = int(num_of_lvls - 2)

    h5_exist = 0
    for i in range(num_of_lvls):
        outfile = f'{str(outdir_parent)}/level_{i}/{input_file.name[:-4]}_bag_features.h5'
        outfile = Path(outfile)
        if outfile.exists():
            h5_exist += 1

    if h5_exist == num_of_lvls:
        return True
    else:
        return False


def collate_features(batch: dict) -> dict:
    img = torch.cat([item["X"] for item in batch], dim=0)
    lbl = np.vstack([item["y"] for item in batch])
    z_level = np.asarray([item["z_lvl"] for item in batch])
    coords = np.vstack([item["coord"] for item in batch])

    return {
        "img": img,
        "label": lbl,
        "z_lvls": z_level,
        "coords": coords
    }


def diffquik_or_papsmear(
        imgdir: Union[PosixPath, str],
        keywords: List[str]
) -> bool:
    if isinstance(imgdir, PosixPath):
        imgdir = str(imgdir)
    for keyword in keywords:
        if keyword in imgdir:
            return True
    return False


def return_cell_value(df: pd.core.frame.DataFrame,
                      find_col: str,
                      find_value: Union[str, int],
                      get_col: str
                      ) -> Union[str, int]:
    get_row_index = df[df[find_col] == find_value].index[0]
    return df.loc[get_row_index, get_col]


def return_df_from_csv(
    filename: Union[PosixPath, str]
) -> pd.core.frame.DataFrame:
    # print(f'File {filename.stem}...')
    if isinstance(filename, PosixPath):
        filename = str(filename)

    if filename.endswith('.tsv'):
        return pd.read_csv(
            filename,
            sep='\t',
            usecols=lambda c: not c.startswith("Unnamed:")
        )
    return pd.read_csv(filename,
                       usecols=lambda c: not c.startswith("Unnamed:"))


def generate_transforms(pretrained: bool) -> transforms.Compose:
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    out_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )

    return out_transforms
