from itertools import chain, cycle
from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import pandas as pd
from matplotlib import pyplot as plt
from pyvips import Image
from torch import DoubleTensor


def return_df_from_csv(filename: Union[Path, str]
                       ) -> pd.core.frame.DataFrame:
    # print(f'File {filename.stem}...')
    if isinstance(filename, Path):
        filename = str(filename)

    if filename.endswith('.tsv'):
        return pd.read_csv(
            filename,
            sep='\t',
            usecols=lambda c: not c.startswith("Unnamed:")
        )
    return pd.read_csv(filename,
                       usecols=lambda c: not c.startswith("Unnamed:"))


def does_h5_exist(input_file: Union[Path, str]) -> bool:
    outdir_parent = Path('/home/asohn3/baraslab/Data/panc_cyto/tiles_384')

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


def save_hdf5(
        output_path: str,
        asset_dict: dict,
        attr_dict: Optional[dict] = None,
        mode: str = 'a') -> str:

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


def get_num_tiles(h5_file: str) -> Tuple[str, int]:
    f = h5py.File(h5_file)
    stem = Path(h5_file).stem
    num_tiles = int(f['tiles'].shape[0])
    return (stem, num_tiles)


def WeightForBalancedDataset(
        splits_csv_file: str,
        num_classes: int
) -> DoubleTensor:

    df = return_df_from_csv(splits_csv_file)

    N = float(len(df))

    if num_classes == 2:
        cls_lbls = [1, 3]
        wt_per_cls = [N/len(df[df['cyto_diag_groups'] == c])
                      for c in cls_lbls]
        weight = [0] * int(N)

        for idx in range(len(df)):
            y = df['cyto_diag_groups'].iloc[idx]
            if y == 1:
                yc = int(0)
            elif y == 3:
                yc = int(1)
            weight[idx] = wt_per_cls[yc]

    elif num_classes == 3:
        cls_lbls = [1, 2, 3]
        wt_per_cls = [N/len(df[df['cyto_diag_groups'] == c])
                      for c in cls_lbls]
        weight = [0] * int(N)

        for idx in range(len(df)):
            y = df['cyto_diag_groups'].iloc[idx]
            weight[idx] = wt_per_cls[int(y-1)]

    return DoubleTensor(weight)


def plot_roc(
        fpr: dict,
        tpr: dict,
        roc_auc: dict,
        n_classes: int,
        savedir: str) -> plt:
    plt.figure()
    lw = 2

    if n_classes == 2:
        plt.plot(
            fpr,
            tpr,
            label="AUC={0:0.2f}".format(roc_auc),
            color="darkorange",
            linestyle=":",
            linewidth=4
        )
    else:
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average (area={0:0.2f})".format(roc_auc["micro"]),
            color="deeppink",
            linestyle=":",
            linewidth=4
        )

        # plt.plot(
        #     fpr["macro"],
        #     tpr["macro"],
        #     label="macro-average (area={0:0.2f})".format(roc_auc["macro"]),
        #     color="navy",
        #     linestyle=":",
        #     linewidth=4
        # )

        colors = cycle(["aqua", "darkorange", "cornflowerblue"])
        for i, color in zip(range(n_classes), colors):
            if i == 0:
                cyto_diag_group = 'no tumor'
            elif i == 1:
                cyto_diag_group = 'atypical/suspicious'
            elif i == 2:
                cyto_diag_group = 'adenocarcinoma'
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=lw,
                label="{0} (area={1:0.2f})".format(
                    cyto_diag_group, roc_auc[i]
                )
            )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Curves")
    plt.legend(loc="lower right")
    return plt.savefig(savedir)


def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error
