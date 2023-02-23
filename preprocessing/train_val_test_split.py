import copy
from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


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


def kfold_train_val_test_split(trainval_dset: Union[pd.DataFrame, str],
                               parent_outdir: str,
                               num_splits: int):
    if isinstance(trainval_dset, str):
        trainval_dset = return_df_from_csv(trainval_dset)

    dset = copy.deepcopy(trainval_dset)
    X = dset.original_filename
    y = dset.cyto_diag_groups
    kf = StratifiedShuffleSplit(
        n_splits=num_splits, test_size=0.2, random_state=7)

    splitted = list(kf.split(X, y))

    for i in range(len(splitted)):
        train_indices = splitted[i][0]
        val_test_indices = splitted[i][1]

        df_train = (copy.deepcopy(dset)).iloc[train_indices]
        df_val_test = (copy.deepcopy(dset)).iloc[val_test_indices]

        kf2 = StratifiedShuffleSplit(
            n_splits=1, test_size=0.5, random_state=7)

        splitted_val_test = list(
            kf2.split(df_val_test.original_filename, df_val_test.cyto_diag_groups))

        val_indices = splitted_val_test[0][0]
        test_indices = splitted_val_test[0][1]

        df_val = (copy.deepcopy(df_val_test)).iloc[val_indices]
        df_test = (copy.deepcopy(df_val_test)).iloc[test_indices]

        df_train.to_csv(
            f'{parent_outdir}/fold_{i+1}_train.tsv', sep='\t', index=False)
        df_val.to_csv(f'{parent_outdir}/fold_{i+1}_val.tsv',
                      sep='\t', index=False)
        df_test.to_csv(f'{parent_outdir}/fold_{i+1}_test.tsv',
                       sep='\t', index=False)
