import argparse
import time
from pathlib import Path, PosixPath
from typing import Union

import cv2
import numpy as np
from joblib import Parallel, delayed

from utils import diffquik_or_papsmear, return_df_from_csv, save_hdf5
from wsi import Slide


class Tiler:
    def __init__(self,
                 cyto_fp: Union[PosixPath, str],
                 get_Z: int,
                 ) -> None:

        if type(cyto_fp) == str:
            self.cyto_fp = Path(cyto_fp)
        else:
            self.cyto_fp = cyto_fp
        self.get_Z = get_Z
        self.load_cytowsi_as_np(get_Z)

    def load_cytowsi_as_np(self,
                           get_Z: int) -> np.ndarray:

        self.cytowsi = np.uint8(
            Slide(self.cyto_fp).vips_2_nparr_z(get_Z)
        )

        return self.cytowsi

    def extract_patches(self,
                        tile_size: int,
                        odir: str) -> None:

        start = time.time()
        rgb = self.cytowsi.copy()
        rgb = self.apply_clahe_img(rgb, self.cyto_fp)
        rgb = self.apply_brightness_contrast(rgb, self.cyto_fp)

        saveas = f'{odir}/level_{self.get_Z}/{self.cyto_fp.stem}_bag.h5'

        _tiles = []
        _coords = []

        mask = self.find_roi_normal(rgb, self.cyto_fp)

        size_img = rgb.shape[:2]
        x = np.arange(0, size_img[1], tile_size)
        y = np.arange(0, size_img[0], tile_size)

        xx, yy = np.meshgrid(x, y)
        loc_list = [
            (xx.flatten()[i], yy.flatten()[i]) for i in range(len(xx.flatten()[:]))
        ]
        locs_arr = np.array(loc_list)

        if tile_size == 256:
            roi_mask_shape = 65536
        elif tile_size == 384:
            roi_mask_shape = 147456

        for each_loc in locs_arr:
            roi_mask = mask[
                int(each_loc[1]): int(each_loc[1]) + int(tile_size),
                int(each_loc[0]): int(each_loc[0]) + int(tile_size)
            ]

            ROI_exist = roi_mask.sum() > 0
            if ROI_exist and (roi_mask.shape[0] * roi_mask.shape[1] == int(roi_mask_shape)):
                _coords.append(
                    [
                        int(each_loc[1]),
                        int(each_loc[1]) + int(tile_size),
                        int(each_loc[0]),
                        int(each_loc[0]) + int(tile_size)
                    ]
                )
                tile = rgb[
                    int(each_loc[1]): int(each_loc[1]) + int(tile_size),
                    int(each_loc[0]): int(each_loc[0]) + int(tile_size)
                ]
                _tiles.append(tile)

        _tiles = np.asarray(_tiles, dtype=np.uint8)
        _coords = np.asarray(_coords, dtype=np.int32)

        asset_dict = {
            'tiles': _tiles,
            'coords': _coords
        }
        save_hdf5(output_path=saveas,
                  asset_dict=asset_dict,
                  attr_dict=None,
                  mode='w')

        print(
            f'Time elapsed for {self.cyto_fp.stem}: {time.time() - start} (num of tiles: {len(_coords)})')

    @staticmethod
    def find_roi_normal(img, fname):
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        is_img_DQ = diffquik_or_papsmear(fname, ['DQ'])

        if is_img_DQ == True:
            lower_blue = np.array([110, 50, 25])
            upper_blue = np.array([130, 255, 255])
        else:
            lower_blue = np.array([94, 50, 15])
            upper_blue = np.array([128, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # (50, 50)
        close_kernel = np.ones((50, 50), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        # (30, 30)
        open_kernel = np.ones((30, 30), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

        return mask

    @staticmethod
    def apply_brightness_contrast(input_img, img_name):
        """
        alpha: contrast control [1.0, 3.0]
        beta: brightness control [0, 100]
        """
        is_img_DQ = diffquik_or_papsmear(img_name, ['DQ'])
        if is_img_DQ:
            alpha = 1.5
            beta = 5
        else:
            alpha = 1.0
            beta = 5
        adjusted = cv2.convertScaleAbs(input_img, alpha=alpha, beta=beta)
        return adjusted

    @staticmethod
    def apply_clahe_img(input_img, img_name):
        is_img_DQ = diffquik_or_papsmear(img_name, ['DQ'])
        if is_img_DQ:
            limit = 0.25
        else:
            limit = 1.0
        size = (8, 8)

        # convert to LAB color space
        # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab = cv2.cvtColor(input_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=size)
        cl = clahe.apply(l)

        # merge the CLAHE enhanced L-channel w/ channels a & b
        l_img = cv2.merge((cl, a, b))

        # convert img back to BGR color space
        # enhanced_img = cv2.cvtColor(l_img, cv2.COLOR_LAB2BGR)
        enhanced_img = cv2.cvtColor(l_img, cv2.COLOR_LAB2RGB)

        return enhanced_img


def save_tiles(input_file, z, tilesize, outdir):
    # annotation_file = '/home/asohn3/baraslab/Data/panc_cyto/final_panc_dx.csv'
    # print(f'Loading file... {input_file.stem}/level_{z}')

    fname = Path(input_file).stem
    save_dir = f'{outdir}/level_{z}'

    save_file = Path(f'{save_dir}/{fname}_bag.h5')
    if save_file.exists():
        print(f'{save_file} Already exists!')
        pass
    else:
        print(f'Loading file... {input_file.stem}/level_{z}')
        tiler = Tiler(
            cyto_fp=input_file,
            get_Z=z,
        )

        tiler.extract_patches(
            tile_size=tilesize,
            odir=outdir
        )


# WSI tessellation settings
parser = argparse.ArgumentParser(
    description="Configurations for WSI tessellation"
)
parser.add_argument(
    '--wsi_dir',
    type=str,
    default=None,
    help='Directory for pancreas fine-needle aspiration whole slide images'
)
parser.add_argument(
    '--out_pdir',
    type=str,
    default=None,
    help='Parent output directory'
)
parser.add_argument(
    '--csv',
    type=str,
    default=None,
    help='CSV file of filenames and labels'
)
parser.add_argument(
    '--z',
    type=int,
    default=6,
    help='Extract tiles from plane Z or all planes if z is None (default: 6)'
)
parser.add_argument(
    '--n_jobs',
    type=int,
    default=10,
    help='Number of cores for parallel processing (default: 10)'
)

args = parser.parse_args()


def main(args):

    input_dir = args.input_dir
    out_pdir = args.out_pdir
    csv_file = args.csv_file
    n_jobs = args.n_jobs
    z = args.z

    original_fnames = return_df_from_csv(csv_file)['original_filename']

    filelist = [
        Path(f'{input_dir}/{file}') for file in original_fnames
    ]

    tsize = args.tile_size
    out_dir = f'{out_pdir}/tiles_{tsize}'

    if z is not None:
        Parallel(n_jobs=n_jobs)(delayed(save_tiles)(fname,
                                                    z,
                                                    tsize,
                                                    out_dir)
                                for fname in filelist)
    else:
        for z in range(11):
            print(f'Number of files at level_{z}... {len(filelist)}')
            Parallel(n_jobs=10)(delayed(save_tiles)(fname,
                                                    z,
                                                    tsize,
                                                    out_dir)
                                for fname in filelist)


if __name__ == '__main__':
    main(args)
